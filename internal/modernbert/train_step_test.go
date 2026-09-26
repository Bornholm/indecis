package modernbert

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/bornholm/indecis/internal/optim"
)

type trainStepFixture struct {
	Texts     []string  `json:"texts"`
	W         []float32 `json:"w"`
	Loss      float64   `json:"loss"`
	TotalNorm float64   `json:"total_norm"`
	LR        float64   `json:"lr"`
	WD        float64   `json:"weight_decay"`
	Clip      float64   `json:"clip"`
	Params    map[string]struct {
		Idx       []int     `json:"idx"`
		Grad      []float64 `json:"grad"`
		GradNorm  float64   `json:"grad_norm"`
		After     []float64 `json:"after"`
		DeltaNorm float64   `json:"delta_norm"`
	} `json:"params"`
}

// Un pas complet (forward, backward, écrêtage, AdamW) comparé à PyTorch,
// fixtures de tools/oracle/train_step_fixtures.py.
//
// Le test modifie les poids : il charge son propre modèle.
func TestTrainStepParity(t *testing.T) {
	dir := bekkoDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, tok := loadBekko(t)
	b, err := os.ReadFile("../../testdata/bekko/train_step.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx trainStepFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	H := m.Cfg.Hidden

	var seqs [][]int32
	for _, s := range fx.Texts {
		seqs = append(seqs, tok.Encode(s))
	}
	batch := NewBatch(seqs, m.Cfg.PadID)
	g := m.EnableGrad()
	s, err := m.Forward(batch)
	if err != nil {
		t.Fatal(err)
	}
	pooled := MeanPool(s.Hidden, batch, H)
	var loss float64
	for i := range pooled {
		loss += float64(pooled[i]) * float64(fx.W[i])
	}
	if math.Abs(loss-fx.Loss) > 1e-4 {
		t.Errorf("perte %v, attendu %v", loss, fx.Loss)
	}
	m.Backward(s, MeanPoolBackward(fx.W, batch, H), g)

	byName := map[string]*Param{}
	for _, p := range m.Params() {
		byName[p.Name] = p
	}
	gradAt := func(p *Param, i int) float64 {
		if p == m.Emb {
			return float64(g.Emb.Rows[int32(i/H)][i%H])
		}
		return float64(p.G[i])
	}
	normOf := func(p *Param) float64 {
		var sq float64
		if p == m.Emb {
			for _, row := range g.Emb.Rows {
				for _, v := range row {
					sq += float64(v) * float64(v)
				}
			}
			return math.Sqrt(sq)
		}
		for _, v := range p.G {
			sq += float64(v) * float64(v)
		}
		return math.Sqrt(sq)
	}

	// Gradients : l'erreur admise est relative à la norme du tenseur, les
	// valeurs individuelles pouvant être proches de zéro.
	for name, e := range fx.Params {
		p := byName[name]
		if p == nil {
			t.Fatalf("paramètre %s absent", name)
		}
		if n := normOf(p); math.Abs(n-e.GradNorm) > 1e-3*e.GradNorm {
			t.Errorf("%s : ‖g‖ = %.6g, attendu %.6g", name, n, e.GradNorm)
		}
		scale := e.GradNorm / math.Sqrt(float64(len(p.W)))
		for k, i := range e.Idx {
			if d := math.Abs(gradAt(p, i) - e.Grad[k]); d > 1e-3*math.Abs(e.Grad[k])+1e-2*scale {
				t.Errorf("%s[%d] : g = %.6g, attendu %.6g", name, i, gradAt(p, i), e.Grad[k])
				break
			}
		}
	}

	// Écrêtage puis AdamW, avec les mêmes groupes que l'oracle.
	var dense []optim.Dense
	for _, p := range m.Params() {
		if p == m.Emb {
			continue
		}
		dense = append(dense, optim.Dense{W: p.W, G: p.G, Decay: len(p.Shape) == 2})
	}
	sparse := []optim.Sparse{{W: m.Emb.W, Width: H, Rows: g.Emb.Rows}}
	if n := optim.ClipGradNorm(fx.Clip, dense, sparse); math.Abs(n-fx.TotalNorm) > 1e-3*fx.TotalNorm {
		t.Errorf("norme totale %.6g, attendu %.6g", n, fx.TotalNorm)
	}
	o := optim.New(optim.Config{LR: fx.LR, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: fx.WD})
	o.Step(fx.LR, dense, sparse)

	// Au premier pas d'Adam, chaque poids bouge d'environ ±lr : l'écart
	// admis est une fraction de ce pas.
	for name, e := range fx.Params {
		p := byName[name]
		for k, i := range e.Idx {
			if d := math.Abs(float64(p.W[i]) - e.After[k]); d > 2e-5 {
				t.Errorf("%s[%d] après le pas : %.7g, attendu %.7g", name, i, p.W[i], e.After[k])
				break
			}
		}
	}
}
