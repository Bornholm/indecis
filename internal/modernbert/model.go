// Package modernbert implémente l'encodeur ModernBERT en Go pur, en
// inférence comme en entraînement.
//
// L'implémentation de référence est celle de transformers
// (modeling_modernbert.py) : pré-normalisation sans biais, attention
// alternant fenêtre globale et fenêtre locale, RoPE, MLP à porte GELU.
// Chaque opération a son forward, qui conserve ce que la rétropropagation
// lira, et son backward écrit à la main : l'architecture est fixe, un
// autograd générique n'apporterait que des allocations.
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

// Config reprend les champs de config.json qui déterminent le calcul.
type Config struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	Intermediate int     `json:"intermediate_size"`
	Vocab        int     `json:"vocab_size"`
	NormEps      float64 `json:"norm_eps"`
	GlobalEvery  int     `json:"global_attn_every_n_layers"`
	// LocalAttention est la largeur totale de la fenêtre locale ; un token
	// voit ses voisins jusqu'à LocalAttention/2 positions de distance.
	LocalAttention int     `json:"local_attention"`
	GlobalTheta    float64 `json:"global_rope_theta"`
	LocalTheta     float64 `json:"local_rope_theta"`
	PadID          int32   `json:"pad_token_id"`

	Activation    string `json:"hidden_activation"`
	AttentionBias bool   `json:"attention_bias"`
	MLPBias       bool   `json:"mlp_bias"`
	NormBias      bool   `json:"norm_bias"`
}

// HeadDim est la dimension d'une tête d'attention.
func (c Config) HeadDim() int { return c.Hidden / c.Heads }

// IsGlobal indique si la couche l voit toute la séquence.
func (c Config) IsGlobal(l int) bool { return l%c.GlobalEvery == 0 }

// Window est la demi-fenêtre des couches locales.
func (c Config) Window() int { return c.LocalAttention / 2 }

func (c Config) validate() error {
	switch {
	case c.Hidden <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.Intermediate <= 0 || c.Vocab <= 0:
		return fmt.Errorf("modernbert: dimensions invalides")
	case c.Hidden%c.Heads != 0 || c.HeadDim()%2 != 0:
		return fmt.Errorf("modernbert: hidden_size %d incompatible avec %d têtes", c.Hidden, c.Heads)
	case c.GlobalEvery <= 0 || c.LocalAttention <= 0:
		return fmt.Errorf("modernbert: fenêtres d'attention invalides")
	case c.Activation != "gelu":
		return fmt.Errorf("modernbert: activation %q non prise en charge", c.Activation)
	case c.AttentionBias || c.MLPBias || c.NormBias:
		return fmt.Errorf("modernbert: les biais ne sont pas pris en charge")
	}
	return nil
}

// Param est un tenseur de poids et, en entraînement, son gradient.
type Param struct {
	Name  string
	Shape []int
	W     []float32
	G     []float32 // nil hors entraînement
}

// Layer regroupe les poids d'une couche de l'encodeur.
type Layer struct {
	AttnNorm *Param // nil pour la couche 0, comme dans la référence
	Wqkv     *Param // [3·H, H]
	Wo       *Param // [H, H]
	MLPNorm  *Param // [H]
	Wi       *Param // [2·I, H] : entrée puis porte
	WoMLP    *Param // [H, I]
}

// Model est un encodeur ModernBERT.
type Model struct {
	Cfg       Config
	Emb       *Param // [V, H]
	EmbNorm   *Param
	Layers    []Layer
	FinalNorm *Param

	ropeMu sync.Mutex
	rope   map[float64]*ropeTable
}

// Load lit config.json et model.safetensors dans dir.
func Load(dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("modernbert: config.json : %w", err)
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

// FromTensors assemble un modèle à partir de tenseurs nommés comme dans
// transformers (préfixe « model. » accepté).
func FromTensors(cfg Config, tensors map[string]safetensors.Tensor) (*Model, error) {
	H, I, V := cfg.Hidden, cfg.Intermediate, cfg.Vocab
	get := func(name string, shape ...int) (*Param, error) {
		t, ok := tensors[name]
		if !ok {
			t, ok = tensors["model."+name]
		}
		if !ok {
			return nil, fmt.Errorf("modernbert: tenseur %s absent", name)
		}
		if fmt.Sprint(t.Shape) != fmt.Sprint(shape) {
			return nil, fmt.Errorf("modernbert: %s : forme %v, attendu %v", name, t.Shape, shape)
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

// Params liste tous les paramètres, dans un ordre stable.
func (m *Model) Params() []*Param {
	ps := []*Param{m.Emb, m.EmbNorm}
	for _, L := range m.Layers {
		if L.AttnNorm != nil {
			ps = append(ps, L.AttnNorm)
		}
		ps = append(ps, L.Wqkv, L.Wo, L.MLPNorm, L.Wi, L.WoMLP)
	}
	return append(ps, m.FinalNorm)
}

// ropeTable contient cos et sin pour chaque position et chaque demi-dimension.
type ropeTable struct {
	n        int
	cos, sin []float32 // [n, D/2]
}

// ropeFor retourne une table couvrant au moins n positions.
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
		// Même arrondi que la référence : fréquences et positions en float32.
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
