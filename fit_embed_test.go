package indecis

import (
	"context"
	"math"
	"testing"
)

// Le gradient de la perte de choix, rétropropagé à travers la
// normalisation et les deux passes, suit les différences finies.
func TestChoiceStepGradient(t *testing.T) {
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	b := ChoiceBatch{
		Texts:      []string{"Ma facture de mars est fausse", "Le serveur ne répond plus", "Réunion jeudi à 10 h ?"},
		Candidates: []Candidate{{Name: "Facturation"}, {Name: "Support technique", Description: "pannes"}, {Name: "Agenda"}, {Name: "Juridique"}},
		Correct:    [][]int{{0}, {1}, {2, 3}},
	}
	grads := m.enc.EnableGrad()
	m.enc.ZeroGrad(grads)
	m.choiceStep(b, DefaultEmbedScale, grads)
	lossAt := func() float64 {
		g := m.enc.EnableGrad()
		m.enc.ZeroGrad(g)
		l, _ := m.choiceStep(b, DefaultEmbedScale, g)
		return l
	}
	w := m.enc.FinalNorm.W
	g := append([]float32(nil), m.enc.FinalNorm.G...)
	checked := 0
	for i := range w {
		if math.Abs(float64(g[i])) < 1e-2 {
			continue
		}
		const eps = 1e-2
		old := w[i]
		w[i] = old + eps
		up := lossAt()
		w[i] = old - eps
		down := lossAt()
		w[i] = old
		num := (up - down) / (2 * eps)
		if math.Abs(num-float64(g[i])) > 0.05*math.Abs(num)+1e-3 {
			t.Errorf("γ[%d] : analytique %.5f, numérique %.5f", i, g[i], num)
		}
		if checked++; checked == 8 {
			break
		}
	}
	if checked == 0 {
		t.Fatal("aucun gradient assez grand pour être vérifié")
	}
}

func TestFitEmbeddingsLearns(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	cands := []Candidate{{Name: "zorblax"}, {Name: "quimpo"}}
	texts := []string{"Ma facture est fausse", "Le serveur est en panne"}
	b := ChoiceBatch{Texts: texts, Candidates: cands, Correct: [][]int{{0}, {1}}}
	opts := DefaultTrainOptions()
	opts.Epochs, opts.LR, opts.Warmup = 30, 2e-4, 0
	if err := m.FitEmbeddings(ctx, []ChoiceBatch{b}, opts); err != nil {
		t.Fatal(err)
	}
	a, err := m.ChooseNearest(ctx, cands, texts...)
	if err != nil {
		t.Fatal(err)
	}
	if a[0].Choice != "zorblax" || a[1].Choice != "quimpo" {
		t.Fatalf("associations arbitraires non apprises : %v, %v", a[0].Probs, a[1].Probs)
	}
}
