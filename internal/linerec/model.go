// Package linerec implements the text-line recognizer of indecis's OCR, in
// pure Go, for inference and training.
//
// The architecture is that of Indra OCR's rec.onnx (see
// tools/oracle/ocr_rec.py, the PyTorch reference): a line resized to a
// fixed height, in a window of three line heights, is cut into vertical
// strips of Stride pixels, one step each. A stem projects the middle band
// of each strip; a context branch pools the whole window over its rows;
// depthwise 1D blocks with squeeze-excitation mix the steps; a head gives
// each step a distribution over the symbols, read with CTC.
//
// Activations are laid out step-major, [steps, channels]: pointwise
// convolutions are then matrix products, and a depthwise convolution runs
// over contiguous channels.
package linerec

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/safetensors"
)

// Config holds the dimensions of a recognizer (config.json).
type Config struct {
	Height      int    `json:"height"` // line height in pixels
	Window      int    `json:"window"` // line heights in the input window
	Stride      int    `json:"stride"` // pixels per step
	Dim         int    `json:"dim"`
	Expand      int    `json:"expand"`
	SE          int    `json:"se"`
	Blocks      int    `json:"blocks"`
	Kernel      int    `json:"kernel"`
	CtxChannels int    `json:"ctx_channels"`
	CtxDim      int    `json:"ctx_dim"`
	CtxHeads    int    `json:"ctx_heads"`
	Vocab       int    `json:"vocab"` // symbols, blank = 0
	Norm        string `json:"norm"`  // stem normalization: "batch" or "layer"
}

// Rows is the height of the input window in pixels.
func (c Config) Rows() int { return c.Window * c.Height }

// CtxRows is the number of rows the context branch pools over.
func (c Config) CtxRows() int { return c.Rows() / c.Stride }

func (c Config) validate() error {
	switch {
	case c.Height <= 0 || c.Window <= 0 || c.Stride <= 0 || c.Dim <= 0 || c.Expand <= 0 ||
		c.SE <= 0 || c.Blocks < 0 || c.Kernel <= 0 || c.CtxChannels <= 0 || c.CtxDim <= 0 ||
		c.CtxHeads <= 0 || c.Vocab < 2:
		return fmt.Errorf("linerec: invalid dimensions")
	case c.Window%2 != 1:
		return fmt.Errorf("linerec: window %d must be odd", c.Window)
	case c.Rows()%c.Stride != 0:
		return fmt.Errorf("linerec: window height %d not a multiple of the stride %d", c.Rows(), c.Stride)
	case c.CtxDim%c.CtxHeads != 0:
		return fmt.Errorf("linerec: ctx_dim %d not a multiple of %d heads", c.CtxDim, c.CtxHeads)
	case c.Kernel%2 != 1:
		return fmt.Errorf("linerec: kernel %d must be odd", c.Kernel)
	case c.Norm != "batch" && c.Norm != "layer":
		return fmt.Errorf("linerec: norm %q not supported", c.Norm)
	}
	return nil
}

// Param is a weight tensor and, during training, its gradient.
type Param struct {
	Name  string
	Shape []int
	W     []float32
	G     []float32 // nil outside training

	// Set by Prepare, for inference: the matrix packed for linalg, or the
	// depthwise kernel transposed to [taps, channels].
	packed *linalg.PackedB
	transp []float32
}

// kernelT returns a depthwise kernel [C, 1, taps…] transposed to [taps, C].
func (p *Param) kernelT(taps, C int) []float32 {
	if p.transp != nil {
		return p.transp
	}
	t := make([]float32, taps*C)
	for c := range C {
		for k := range taps {
			t[k*C+c] = p.W[c*taps+k]
		}
	}
	return t
}

// Prepare packs the weights for inference. The model must not be trained
// afterwards (or Unprepare first): the packed copies would be stale.
func (m *Model) Prepare() {
	c := m.Cfg
	pack := func(p *Param, out, k int) { p.packed = linalg.PackB(p.W, k, out, true) }
	pack(m.Stem, c.Dim, c.Height*c.Stride)
	pack(m.CtxPatch, c.CtxChannels, c.Stride*c.Stride)
	pack(m.CtxPW, c.CtxDim, c.CtxChannels)
	pack(m.CtxScore, c.CtxHeads, c.CtxDim)
	pack(m.CtxProj, c.Dim, c.CtxDim)
	m.CtxDW.transp = nil
	m.CtxDW.transp = m.CtxDW.kernelT(9, c.CtxChannels)
	for i := range m.Blocks {
		b := &m.Blocks[i]
		pack(b.Expand, c.Expand, c.Dim)
		pack(b.Project, c.Dim, c.Expand)
		b.DW.transp = nil
		b.DW.transp = b.DW.kernelT(c.Kernel, c.Dim)
	}
	pack(m.Head, c.Vocab, c.Dim)
}

// Unprepare drops the packed weights.
func (m *Model) Unprepare() {
	for _, p := range m.Params() {
		p.packed, p.transp = nil, nil
	}
}

// Block groups the weights of a depthwise block.
type Block struct {
	DW, DWBias         *Param // [Dim, 1, Kernel], [Dim]
	Expand, ExpandBias *Param // [Expand, Dim, 1]
	FC1, FC1Bias       *Param // [SE, Expand, 1]
	FC2, FC2Bias       *Param // [Expand, SE, 1]
	Project, ProjBias  *Param // [Dim, Expand, 1]
}

// Model is a line recognizer.
type Model struct {
	Cfg Config

	Stem, StemBias *Param // [Dim, 1, Height, Stride]

	CtxPatch, CtxPatchBias *Param // [CtxChannels, 1, Stride, Stride]
	CtxDW                  *Param // [CtxChannels, 1, 3, 3], no bias
	CtxPW, CtxPWBias       *Param // [CtxDim, CtxChannels, 1, 1]
	CtxScore, CtxScoreBias *Param // [CtxHeads, CtxDim, 1, 1]
	RowPrior               *Param // [1, CtxHeads, CtxRows, 1]
	CtxProj, CtxProjBias   *Param // [Dim, CtxDim, 1]

	// Stem normalization: NormW and NormB are the affine weights of the
	// batch or layer norm; RunMean and RunVar the batch norm's statistics
	// (nil for a layer norm).
	NormW, NormB    *Param
	RunMean, RunVar *Param

	Blocks []Block

	Head, HeadBias *Param // [Vocab, Dim, 1]
}

// normEps is PyTorch's default for BatchNorm1d and LayerNorm.
const normEps = 1e-5

// Load reads config.json and model.safetensors from dir.
func Load(dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("linerec: config.json: %w", err)
	}
	tensors, _, err := safetensors.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	return FromTensors(cfg, tensors)
}

// FromTensors assembles a model from tensors named as in the PyTorch
// reference.
func FromTensors(cfg Config, tensors map[string]safetensors.Tensor) (*Model, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	var err error
	get := func(name string, shape ...int) *Param {
		if err != nil {
			return nil
		}
		t, ok := tensors[name]
		if !ok {
			err = fmt.Errorf("linerec: tensor %s missing", name)
			return nil
		}
		if fmt.Sprint(t.Shape) != fmt.Sprint(shape) {
			err = fmt.Errorf("linerec: %s: shape %v, expected %v", name, t.Shape, shape)
			return nil
		}
		return &Param{Name: name, Shape: shape, W: t.Data}
	}

	c := cfg
	m := &Model{Cfg: cfg}
	m.Stem = get("stem.weight", c.Dim, 1, c.Height, c.Stride)
	m.StemBias = get("stem.bias", c.Dim)
	m.CtxPatch = get("context.patch.weight", c.CtxChannels, 1, c.Stride, c.Stride)
	m.CtxPatchBias = get("context.patch.bias", c.CtxChannels)
	m.CtxDW = get("context.dw.weight", c.CtxChannels, 1, 3, 3)
	m.CtxPW = get("context.pw.weight", c.CtxDim, c.CtxChannels, 1, 1)
	m.CtxPWBias = get("context.pw.bias", c.CtxDim)
	m.CtxScore = get("context.score.weight", c.CtxHeads, c.CtxDim, 1, 1)
	m.CtxScoreBias = get("context.score.bias", c.CtxHeads)
	m.RowPrior = get("context.row_prior", 1, c.CtxHeads, c.CtxRows(), 1)
	m.CtxProj = get("context.proj.weight", c.Dim, c.CtxDim, 1)
	m.CtxProjBias = get("context.proj.bias", c.Dim)
	if c.Norm == "batch" {
		m.NormW = get("stem_bn.weight", c.Dim)
		m.NormB = get("stem_bn.bias", c.Dim)
		m.RunMean = get("stem_bn.running_mean", c.Dim)
		m.RunVar = get("stem_bn.running_var", c.Dim)
	} else {
		m.NormW = get("stem_ln.weight", c.Dim)
		m.NormB = get("stem_ln.bias", c.Dim)
	}
	m.Blocks = make([]Block, c.Blocks)
	for i := range m.Blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		b := &m.Blocks[i]
		b.DW = get(p+"dw.weight", c.Dim, 1, c.Kernel)
		b.DWBias = get(p+"dw.bias", c.Dim)
		b.Expand = get(p+"expand.weight", c.Expand, c.Dim, 1)
		b.ExpandBias = get(p+"expand.bias", c.Expand)
		b.FC1 = get(p+"se.fc1.weight", c.SE, c.Expand, 1)
		b.FC1Bias = get(p+"se.fc1.bias", c.SE)
		b.FC2 = get(p+"se.fc2.weight", c.Expand, c.SE, 1)
		b.FC2Bias = get(p+"se.fc2.bias", c.Expand)
		b.Project = get(p+"project.weight", c.Dim, c.Expand, 1)
		b.ProjBias = get(p+"project.bias", c.Dim)
	}
	m.Head = get("head.weight", c.Vocab, c.Dim, 1)
	m.HeadBias = get("head.bias", c.Vocab)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Params lists the trainable parameters, in a stable order. The batch
// norm's running statistics are not among them.
func (m *Model) Params() []*Param {
	ps := []*Param{
		m.Stem, m.StemBias,
		m.CtxPatch, m.CtxPatchBias, m.CtxDW, m.CtxPW, m.CtxPWBias,
		m.CtxScore, m.CtxScoreBias, m.RowPrior, m.CtxProj, m.CtxProjBias,
		m.NormW, m.NormB,
	}
	for _, b := range m.Blocks {
		ps = append(ps, b.DW, b.DWBias, b.Expand, b.ExpandBias, b.FC1, b.FC1Bias,
			b.FC2, b.FC2Bias, b.Project, b.ProjBias)
	}
	return append(ps, m.Head, m.HeadBias)
}

// Tensors returns all weights under their reference names, running
// statistics included, for writing.
func (m *Model) Tensors() map[string]safetensors.Tensor {
	out := map[string]safetensors.Tensor{}
	ps := m.Params()
	if m.RunMean != nil {
		ps = append(ps, m.RunMean, m.RunVar)
	}
	for _, p := range ps {
		out[p.Name] = safetensors.Tensor{Shape: p.Shape, Data: p.W}
	}
	return out
}

// Steps returns the padded and valid step counts of a line of width w
// pixels: the line is padded to a multiple of 2·Stride, as the reference
// reads a line alone.
func (c Config) Steps(w int) (padded, valid int) {
	m := 2 * c.Stride
	wp := (w + m - 1) / m * m
	return wp / c.Stride, (w + c.Stride - 1) / c.Stride
}
