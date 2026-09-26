package modernbert

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/bornholm/indecis/internal/safetensors"
)

// tinyModel construit un modèle aléatoire assez petit pour une vérification
// exhaustive par différences finies, mais qui exerce tout : fenêtre locale
// plus courte que les séquences, couche 0 sans attn_norm, plusieurs têtes.
func tinyModel(t *testing.T, seed int64) *Model {
	t.Helper()
	cfg := Config{
		Hidden: 8, Layers: 4, Heads: 2, Intermediate: 6, Vocab: 17,
		NormEps: 1e-5, GlobalEvery: 3, LocalAttention: 4,
		GlobalTheta: 10000, LocalTheta: 500, Activation: "gelu",
	}
	r := rand.New(rand.NewSource(seed))
	rnd := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(r.NormFloat64() * 0.5)
		}
		return safetensors.Tensor{Shape: shape, Data: data}
	}
	norm := func() safetensors.Tensor {
		t := rnd(cfg.Hidden)
		for i := range t.Data {
			t.Data[i] += 1
		}
		return t
	}
	H, I := cfg.Hidden, cfg.Intermediate
	ts := map[string]safetensors.Tensor{
		"embeddings.tok_embeddings.weight": rnd(cfg.Vocab, H),
		"embeddings.norm.weight":           norm(),
		"final_norm.weight":                norm(),
	}
	for l := 0; l < cfg.Layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		if l > 0 {
			ts[p+"attn_norm.weight"] = norm()
		}
		ts[p+"attn.Wqkv.weight"] = rnd(3*H, H)
		ts[p+"attn.Wo.weight"] = rnd(H, H)
		ts[p+"mlp_norm.weight"] = norm()
		ts[p+"mlp.Wi.weight"] = rnd(2*I, H)
		ts[p+"mlp.Wo.weight"] = rnd(H, I)
	}
	m, err := FromTensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// lossOf calcule L = Σ_b pooled_b · w_b en float64.
func lossOf(t *testing.T, m *Model, b Batch, w []float32) float64 {
	t.Helper()
	s, err := m.Forward(b)
	if err != nil {
		t.Fatal(err)
	}
	pooled := MeanPool(s.Hidden, b, m.Cfg.Hidden)
	var l float64
	for i := range pooled {
		l += float64(pooled[i]) * float64(w[i])
	}
	return l
}

func TestBackwardFiniteDifferences(t *testing.T) {
	m := tinyModel(t, 1)
	H := m.Cfg.Hidden
	// Séquences plus longues que la demi-fenêtre (2) et de longueurs
	// différentes : fenêtre locale et padding sont exercés.
	b := NewBatch([][]int32{{1, 5, 9, 3, 3, 12, 7}, {2, 16, 4, 0, 8}}, 0)
	r := rand.New(rand.NewSource(2))
	w := make([]float32, b.B()*H)
	for i := range w {
		w[i] = float32(r.NormFloat64())
	}

	g := m.EnableGrad()
	s, err := m.Forward(b)
	if err != nil {
		t.Fatal(err)
	}
	m.Backward(s, MeanPoolBackward(w, b, H), g)

	// Différence centrée extrapolée (Richardson) : erreur en O(ε⁴), sans
	// descendre à un pas où le bruit float32 dominerait.
	const eps = 1e-2
	central := func(param []float32, i int, h float32) float64 {
		orig := param[i]
		param[i] = orig + h
		lp := lossOf(t, m, b, w)
		param[i] = orig - h
		lm := lossOf(t, m, b, w)
		param[i] = orig
		return (lp - lm) / (2 * float64(h))
	}
	check := func(name string, i int, param []float32, analytic float64) {
		fd := (4*central(param, i, eps/2) - central(param, i, eps)) / 3
		if math.Abs(analytic-fd) > 2e-3+2e-2*math.Abs(fd) {
			t.Errorf("%s[%d] : analytique %.6g, différences finies %.6g", name, i, analytic, fd)
		}
	}

	checked := 0
	for _, p := range m.Params() {
		if p == m.Emb {
			continue
		}
		for i := range p.W {
			check(p.Name, i, p.W, float64(p.G[i]))
			checked++
		}
	}
	for id := int32(0); id < int32(m.Cfg.Vocab); id++ {
		row := g.Emb.Rows[id]
		for j := 0; j < H; j++ {
			var analytic float64
			if row != nil {
				analytic = float64(row[j])
			}
			check(fmt.Sprintf("emb[%d]", id), int(id)*H+j, m.Emb.W, analytic)
			checked++
		}
	}
	t.Logf("%d paramètres vérifiés", checked)
}

// Deux backward successifs sans ZeroGrad doivent doubler les gradients :
// l'accumulation sur plusieurs lots en dépend.
func TestBackwardAccumulates(t *testing.T) {
	m := tinyModel(t, 3)
	H := m.Cfg.Hidden
	b := NewBatch([][]int32{{1, 2, 3, 4}}, 0)
	w := make([]float32, H)
	for i := range w {
		w[i] = 1
	}
	g := m.EnableGrad()
	s, _ := m.Forward(b)
	m.Backward(s, MeanPoolBackward(w, b, H), g)
	once := append([]float32(nil), m.Layers[2].Wi.G...)
	emb := append([]float32(nil), g.Emb.Rows[3]...)
	m.Backward(s, MeanPoolBackward(w, b, H), g)
	for i, v := range m.Layers[2].Wi.G {
		if math.Abs(float64(v-2*once[i])) > 1e-5 {
			t.Fatalf("Wi.G[%d] = %v, attendu %v", i, v, 2*once[i])
		}
	}
	for i, v := range g.Emb.Rows[3] {
		if math.Abs(float64(v-2*emb[i])) > 1e-5 {
			t.Fatalf("emb[3][%d] = %v, attendu %v", i, v, 2*emb[i])
		}
	}
	m.ZeroGrad(g)
	if len(g.Emb.Rows) != 0 || m.Layers[2].Wi.G[0] != 0 {
		t.Fatal("ZeroGrad incomplet")
	}
}
