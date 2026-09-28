// Package modernbert implements the ModernBERT encoder in pure Go, for
// both inference and training.
//
// The reference implementation is that of transformers
// (modeling_modernbert.py): bias-free pre-normalization, attention
// alternating global and local windows, RoPE, gated GELU MLP. Each
// operation has its forward pass, which keeps what the backward pass will
// read, and its hand-written backward pass: the architecture is fixed, a
// generic autograd would only add allocations.
package modernbert

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/bornholm/indecis/internal/safetensors"
)

// Config holds the fields of config.json that determine the computation.
type Config struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	Intermediate int     `json:"intermediate_size"`
	Vocab        int     `json:"vocab_size"`
	NormEps      float64 `json:"norm_eps"`
	GlobalEvery  int     `json:"global_attn_every_n_layers"`
	// LocalAttention is the total width of the local window; a token sees
	// its neighbors up to LocalAttention/2 positions away.
	LocalAttention int     `json:"local_attention"`
	GlobalTheta    float64 `json:"global_rope_theta"`
	LocalTheta     float64 `json:"local_rope_theta"`
	PadID          int32   `json:"pad_token_id"`

	Activation    string `json:"hidden_activation"`
	AttentionBias bool   `json:"attention_bias"`
	MLPBias       bool   `json:"mlp_bias"`
	NormBias      bool   `json:"norm_bias"`
}

// HeadDim is the dimension of an attention head.
func (c Config) HeadDim() int { return c.Hidden / c.Heads }

// IsGlobal reports whether layer l sees the whole sequence.
func (c Config) IsGlobal(l int) bool { return l%c.GlobalEvery == 0 }

// Window is the half-window of the local layers.
func (c Config) Window() int { return c.LocalAttention / 2 }

func (c Config) validate() error {
	switch {
	case c.Hidden <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.Intermediate <= 0 || c.Vocab <= 0:
		return fmt.Errorf("modernbert: invalid dimensions")
	case c.Hidden%c.Heads != 0 || c.HeadDim()%2 != 0:
		return fmt.Errorf("modernbert: hidden_size %d incompatible with %d heads", c.Hidden, c.Heads)
	case c.GlobalEvery <= 0 || c.LocalAttention <= 0:
		return fmt.Errorf("modernbert: invalid attention windows")
	case c.Activation != "gelu":
		return fmt.Errorf("modernbert: activation %q not supported", c.Activation)
	case c.AttentionBias || c.MLPBias || c.NormBias:
		return fmt.Errorf("modernbert: biases are not supported")
	}
	return nil
}

// Param is a weight tensor and, during training, its gradient.
type Param struct {
	Name  string
	Shape []int
	W     []float32
	G     []float32 // nil outside training
}

// Layer groups the weights of an encoder layer.
type Layer struct {
	AttnNorm *Param // nil for layer 0, as in the reference
	Wqkv     *Param // [3·H, H]
	Wo       *Param // [H, H]
	MLPNorm  *Param // [H]
	Wi       *Param // [2·I, H]: input then gate
	WoMLP    *Param // [H, I]
}

// Model is a ModernBERT encoder.
type Model struct {
	Cfg       Config
	Emb       *Param // [V, H]
	EmbNorm   *Param
	Layers    []Layer
	FinalNorm *Param

	ropeMu sync.Mutex
	rope   map[float64]*ropeTable

	// embTable replaces Emb.W when the table is read on demand.
	embTable EmbeddingTable
	// packed holds the weights packed for Encode, nil if they need to be
	// redone.
	packMu  sync.Mutex
	packed  []packedLayer
	compact bool         // see SetCompact
	source  WeightSource // see SetCompact
	int8    bool         // see SetInt8
}

// Load reads config.json and model.safetensors from dir.
func Load(dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("modernbert: config.json: %w", err)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	tensors, _, err := safetensors.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	return FromTensors(cfg, tensors)
}

// FromTensors assembles a model from tensors named as in transformers
// (the "model." prefix is accepted).
func FromTensors(cfg Config, tensors map[string]safetensors.Tensor) (*Model, error) {
	H, I, V := cfg.Hidden, cfg.Intermediate, cfg.Vocab
	get := func(name string, shape ...int) (*Param, error) {
		t, ok := tensors[name]
		if !ok {
			t, ok = tensors["model."+name]
		}
		if !ok {
			return nil, fmt.Errorf("modernbert: tensor %s missing", name)
		}
		if fmt.Sprint(t.Shape) != fmt.Sprint(shape) {
			return nil, fmt.Errorf("modernbert: %s: shape %v, expected %v", name, t.Shape, shape)
		}
		return &Param{Name: name, Shape: shape, W: t.Data}, nil
	}

	m := &Model{Cfg: cfg, rope: map[float64]*ropeTable{}}
	var err error
	if m.Emb, err = get("embeddings.tok_embeddings.weight", V, H); err != nil {
		return nil, err
	}
	if m.EmbNorm, err = get("embeddings.norm.weight", H); err != nil {
		return nil, err
	}
	if m.FinalNorm, err = get("final_norm.weight", H); err != nil {
		return nil, err
	}
	m.Layers = make([]Layer, cfg.Layers)
	for l := range m.Layers {
		p := fmt.Sprintf("layers.%d.", l)
		L := &m.Layers[l]
		if l > 0 {
			if L.AttnNorm, err = get(p+"attn_norm.weight", H); err != nil {
				return nil, err
			}
		}
		if L.Wqkv, err = get(p+"attn.Wqkv.weight", 3*H, H); err != nil {
			return nil, err
		}
		if L.Wo, err = get(p+"attn.Wo.weight", H, H); err != nil {
			return nil, err
		}
		if L.MLPNorm, err = get(p+"mlp_norm.weight", H); err != nil {
			return nil, err
		}
		if L.Wi, err = get(p+"mlp.Wi.weight", 2*I, H); err != nil {
			return nil, err
		}
		if L.WoMLP, err = get(p+"mlp.Wo.weight", H, I); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// Params lists all parameters, in a stable order.
func (m *Model) Params() []*Param {
	m.restoreWeights()
	ps := []*Param{m.Emb, m.EmbNorm}
	for _, L := range m.Layers {
		if L.AttnNorm != nil {
			ps = append(ps, L.AttnNorm)
		}
		ps = append(ps, L.Wqkv, L.Wo, L.MLPNorm, L.Wi, L.WoMLP)
	}
	return append(ps, m.FinalNorm)
}

// ropeTable holds cos and sin for each position and each half-dimension.
type ropeTable struct {
	n        int
	cos, sin []float32 // [n, D/2]
}

// ropeFor returns a table covering at least n positions.
func (m *Model) ropeFor(theta float64, n int) *ropeTable {
	m.ropeMu.Lock()
	defer m.ropeMu.Unlock()
	if t := m.rope[theta]; t != nil && t.n >= n {
		return t
	}
	D := m.Cfg.HeadDim()
	half := D / 2
	size := max(n, 512)
	t := &ropeTable{n: size, cos: make([]float32, size*half), sin: make([]float32, size*half)}
	for i := 0; i < half; i++ {
		// Same rounding as the reference: frequencies and positions in float32.
		inv := float32(1 / math.Pow(theta, float64(2*i)/float64(D)))
		for p := 0; p < size; p++ {
			f := float64(float32(p) * inv)
			t.cos[p*half+i] = float32(math.Cos(f))
			t.sin[p*half+i] = float32(math.Sin(f))
		}
	}
	m.rope[theta] = t
	return t
}
