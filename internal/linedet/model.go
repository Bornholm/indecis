// Package linedet implements the text-line detector of indecis's OCR, in
// pure Go, for inference and training.
//
// The architecture follows Indra OCR's det.onnx, read from its graph: a
// stride-2 stem, four stages of MobileNet blocks (strides 4 to 32), an FPN
// merged at stride 4, a shortcut from the stem, and a head giving three
// maps at stride 2: text, picture (logits) and line height (through a
// softplus).
package linedet

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/bornholm/indecis/internal/convnet"
	"github.com/bornholm/indecis/internal/optim"
	"github.com/bornholm/indecis/internal/safetensors"
)

// Stage is one resolution of the backbone.
type Stage struct {
	C      int `json:"c"`      // channels
	K      int `json:"k"`      // depthwise kernel
	Blocks int `json:"blocks"` // residual blocks after the downsampling one
	Expand int `json:"expand"` // channels inside a block
}

// Config holds a detector's dimensions (config.json).
type Config struct {
	Stem    int     `json:"stem"`     // stem channels
	StemLat int     `json:"stem_lat"` // stem shortcut channels, before space-to-depth
	FPN     int     `json:"fpn"`      // channels of the pyramid and the head
	Stages  []Stage `json:"stages"`
	// HeightScale converts the softplus of the height map to pixels of the
	// output map.
	HeightScale float32 `json:"height_scale"`
}

// Base is the configuration of Indra OCR's detector (91 k parameters).
func Base() Config {
	return Config{
		Stem: 16, StemLat: 4, FPN: 24, HeightScale: 8,
		Stages: []Stage{{24, 3, 1, 48}, {32, 3, 2, 64}, {48, 5, 2, 96}, {64, 5, 2, 128}},
	}
}

// Multiple is the side every page is padded to a multiple of: the
// strides of the four stages.
const Multiple = 32

// Param is a weight tensor.
type Param struct {
	Name  string
	Shape []int
	W     []float32
}

// Model is a line detector.
type Model struct {
	Cfg    Config
	params []*Param
	byName map[string]*Param
}

func (m *Model) p(name string) *Param { return m.byName[name] }

// spec lists the parameters of a configuration, with their fan-in.
func spec(c Config) (names []string, shapes [][]int) {
	add := func(name string, shape ...int) {
		names = append(names, name+".weight", name+".bias")
		shapes = append(shapes, shape, shape[:1])
	}
	add("stem", c.Stem, 1, 3, 3)
	add("stem_lat", c.StemLat, c.Stem, 1, 1)
	cin := c.Stem
	for i, s := range c.Stages {
		p := fmt.Sprintf("stages.%d.", i)
		add(p+"down.dw", cin, 1, s.K, s.K)
		add(p+"down.pw", s.C, cin, 1, 1)
		for j := range s.Blocks {
			q := fmt.Sprintf("%sblocks.%d.", p, j)
			add(q+"expand", s.Expand, s.C, 1, 1)
			add(q+"dw", s.Expand, 1, s.K, s.K)
			add(q+"project", s.C, s.Expand, 1, 1)
		}
		add(fmt.Sprintf("lat.%d", i), c.FPN, s.C, 1, 1)
		cin = s.C
	}
	add("smooth", c.FPN, 1, 3, 3)
	add("head.0", c.FPN, c.FPN+4*c.StemLat, 1, 1)
	add("head.1", c.FPN, 1, 3, 3)
	add("head.2", 12, c.FPN, 1, 1)
	return names, shapes
}

// New creates a detector with fresh weights (PyTorch's default
// initialization: uniform in ±1/√fan_in).
func New(cfg Config, seed int64) *Model {
	r := rand.New(rand.NewSource(seed))
	names, shapes := spec(cfg)
	m := &Model{Cfg: cfg, byName: map[string]*Param{}}
	var fanIn int
	for i, name := range names {
		shape := shapes[i]
		n := 1
		for _, d := range shape {
			n *= d
		}
		if len(shape) > 1 {
			fanIn = n / shape[0]
		}
		bound := 1 / math.Sqrt(float64(fanIn))
		w := make([]float32, n)
		for j := range w {
			w[j] = float32((2*r.Float64() - 1) * bound)
		}
		p := &Param{Name: name, Shape: shape, W: w}
		m.params = append(m.params, p)
		m.byName[name] = p
	}
	return m
}

// Load reads config.json and model.safetensors from dir.
func Load(dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("linedet: config.json: %w", err)
	}
	tensors, _, err := safetensors.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	m := New(cfg, 0)
	for _, p := range m.params {
		t, ok := tensors[p.Name]
		if !ok || len(t.Data) != len(p.W) {
			return nil, fmt.Errorf("linedet: tensor %s missing or misshapen", p.Name)
		}
		p.W = t.Data
	}
	return m, nil
}

// Tensors returns the weights for writing.
func (m *Model) Tensors() map[string]safetensors.Tensor {
	out := map[string]safetensors.Tensor{}
	for _, p := range m.params {
		out[p.Name] = safetensors.Tensor{Shape: p.Shape, Data: p.W}
	}
	return out
}

// Params lists the parameters in a stable order.
func (m *Model) Params() []*Param { return m.params }

// Grads holds a gradient buffer per parameter.
type Grads struct {
	G   [][]float32
	idx map[*Param]int
}

// NewGrads allocates zero gradients.
func (m *Model) NewGrads() *Grads {
	g := &Grads{G: make([][]float32, len(m.params)), idx: map[*Param]int{}}
	for i, p := range m.params {
		g.G[i] = make([]float32, len(p.W))
		g.idx[p] = i
	}
	return g
}

func (g *Grads) of(p *Param) []float32 { return g.G[g.idx[p]] }

// Zero resets the gradients.
func (g *Grads) Zero() {
	for _, b := range g.G {
		clear(b)
	}
}

// Add adds o's gradients to g.
func (g *Grads) Add(o *Grads) {
	for i, b := range g.G {
		for j, v := range o.G[i] {
			b[j] += v
		}
	}
}

// Dense pairs the parameters with gradients for optim.AdamW: weight decay
// on the kernels only.
func (m *Model) Dense(g *Grads) []optim.Dense {
	out := make([]optim.Dense, len(m.params))
	for i, p := range m.params {
		out[i] = optim.Dense{W: p.W, G: g.G[i], Decay: len(p.Shape) > 1}
	}
	return out
}

// Forward runs the detector on a page: ink [H, W] (0 = paper, 1 = ink),
// H and W multiples of 32. It returns the output maps [H/2, W/2, 3]:
// text and picture logits, and the raw height (see Height). With train,
// the tape keeps what Backward needs.
func (m *Model) Forward(t *Tape, ink []float32, H, W int, train bool) *Tensor {
	if H%Multiple != 0 || W%Multiple != 0 {
		panic("linedet: page sides must be multiples of 32")
	}
	t.reset(train)
	c := m.Cfg
	x := t.tensor(convnet.Shape{H: H, W: W, C: 1})
	copy(x.V, ink[:H*W])

	stem := t.hs(t.conv(x, m.p("stem.weight"), m.p("stem.bias"), 3, 2, 1, false))
	lat0 := t.s2d(t.hs(t.pw(stem, m.p("stem_lat.weight"), m.p("stem_lat.bias"))))

	h := stem
	var lats []*Tensor
	for i, s := range c.Stages {
		p := fmt.Sprintf("stages.%d.", i)
		h = t.hs(t.dw(h, m.p(p+"down.dw.weight"), m.p(p+"down.dw.bias"), s.K, 2))
		h = t.hs(t.pw(h, m.p(p+"down.pw.weight"), m.p(p+"down.pw.bias")))
		for j := range s.Blocks {
			q := fmt.Sprintf("%sblocks.%d.", p, j)
			e := t.hs(t.pw(h, m.p(q+"expand.weight"), m.p(q+"expand.bias")))
			e = t.hs(t.dw(e, m.p(q+"dw.weight"), m.p(q+"dw.bias"), s.K, 1))
			h = t.add(h, t.pw(e, m.p(q+"project.weight"), m.p(q+"project.bias")))
		}
		l := fmt.Sprintf("lat.%d.", i)
		lats = append(lats, t.hs(t.pw(h, m.p(l+"weight"), m.p(l+"bias"))))
	}
	// Top-down pyramid, down to stride 4.
	f := lats[len(lats)-1]
	for i := len(lats) - 2; i >= 0; i-- {
		f = t.add(lats[i], t.up2(f))
	}
	f = t.hs(t.dw(f, m.p("smooth.weight"), m.p("smooth.bias"), 3, 1))
	f = t.concat(f, lat0)
	f = t.hs(t.pw(f, m.p("head.0.weight"), m.p("head.0.bias")))
	f = t.hs(t.dw(f, m.p("head.1.weight"), m.p("head.1.bias"), 3, 1))
	f = t.pw(f, m.p("head.2.weight"), m.p("head.2.bias"))
	return t.d2s(f)
}

// Backward backpropagates dOut, the gradient of the loss with respect to
// the maps Forward returned, and accumulates the parameter gradients.
func (m *Model) Backward(t *Tape, out *Tensor, dOut []float32, g *Grads) {
	copy(grad(out), dOut)
	for i := len(t.steps) - 1; i >= 0; i-- {
		t.steps[i](g)
	}
}

// Height converts a raw height value of the output maps to output pixels.
func (m *Model) Height(raw float32) float32 {
	if raw > 20 {
		return raw * m.Cfg.HeightScale
	}
	return float32(math.Log1p(math.Exp(float64(raw)))) * m.Cfg.HeightScale
}
