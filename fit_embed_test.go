package indecis

import (
	"context"
	"math"
	"testing"
)

// The gradient of the choice loss, backpropagated through the
// normalization and the two passes, follows the finite differences.
func TestChoiceStepGradient(t *testing.T) {
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	b := ChoiceBatch{
		Texts:      []string{"Ma facture de mars est fausse", "Le serveur ne répond plus", "Réunion jeudi à 10 h ?", "Facture payée deux fois"},
		Candidates: []Candidate{{Name: "Facturation"}, {Name: "Support technique", Description: "pannes"}, {Name: "Agenda"}, {Name: "Juridique"}},
		Correct:    [][]int{{0}, {1}, {2, 3}, {0}}, // two texts for Facturation
	}
	for _, symmetric := range []bool{false, true} {
		checkChoiceGradient(t, m, b, symmetric, nil)
	}
}

// The distillation terms of FitEmbeddings (scores and alignment) follow
// the finite differences too, alone and mixed with the contrastive loss.
func TestChoiceStepDistillGradient(t *testing.T) {
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	teacher, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	// A teacher that differs from the model: its final norm perturbed.
	for i := range teacher.enc.FinalNorm.W {
		teacher.enc.FinalNorm.W[i] *= 1 + 0.5*float32(math.Sin(float64(i)))
	}
	if err := m.checkTeacherEncoder(teacher); err != nil {
		t.Fatal(err)
	}
	b := ChoiceBatch{
		Texts:      []string{"Ma facture de mars est fausse", "Le serveur ne répond plus", "Réunion jeudi à 10 h ?", "Facture payée deux fois"},
		Candidates: []Candidate{{Name: "Facturation"}, {Name: "Support technique", Description: "pannes"}, {Name: "Agenda"}, {Name: "Juridique"}},
		Correct:    [][]int{{0}, {1}, {2, 3}, {0}},
	}
	te, _ := teacher.Embed(context.Background(), b.Texts[0])
	me, _ := m.Embed(context.Background(), b.Texts[0])
	var cos float64
	for k := range te[0] {
		cos += float64(te[0][k]) * float64(me[0][k])
	}
	if cos > 0.999 {
		t.Fatalf("cosine %.5f: the teacher does not differ from the model", cos)
	}
	for _, w := range []struct{ kl, temp, align float64 }{{1, 2, 0}, {0, 1, 1}, {0.5, 2, 1}} {
		tc := newEmbedTeacher(teacher, w.kl, w.temp, w.align)
		checkChoiceGradient(t, m, b, false, tc)
	}
}

func checkChoiceGradient(t *testing.T, m *Model, b ChoiceBatch, symmetric bool, tc *embedTeacher) {
	grads := m.enc.EnableGrad()
	m.enc.ZeroGrad(grads)
	if _, _, err := m.choiceStep(b, DefaultEmbedScale, symmetric, tc, grads); err != nil {
		t.Fatal(err)
	}
	lossAt := func() float64 {
		g := m.enc.EnableGrad()
		m.enc.ZeroGrad(g)
		l, _, err := m.choiceStep(b, DefaultEmbedScale, symmetric, tc, g)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	w := m.enc.FinalNorm.W
	g := append([]float32(nil), m.enc.FinalNorm.G...)
	checked := 0
	for i := range w {
		if math.Abs(float64(g[i])) < 2e-3 {
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
		if math.Abs(num-float64(g[i])) > 0.05*math.Abs(num)+2e-4 {
			t.Errorf("symmetric=%v, distillation %v, gamma[%d]: analytic %.5f, numeric %.5f", symmetric, tc != nil, i, g[i], num)
		}
		if checked++; checked == 8 {
			break
		}
	}
	if checked == 0 {
		t.Fatal("no gradient large enough to check")
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
		t.Fatalf("arbitrary associations not learned: %v, %v", a[0].Probs, a[1].Probs)
	}
}

// Examples move the prototype of an option: an option with an arbitrary
// name becomes the right answer thanks to its examples.
func TestChooseNearestExamples(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	cands := []Candidate{
		{Name: "K1", Examples: []string{"Votre facture de mars est disponible", "Relance : paiement en retard"}},
		{Name: "K2", Examples: []string{"Le serveur de fichiers est en panne", "Impossible de me connecter au VPN"}},
	}
	set, err := m.PrepareCandidates(ctx, cands)
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.ChooseIn(ctx, set, "Pouvez-vous m'envoyer la facture corrigée ?", "Mon ordinateur ne démarre plus")
	if err != nil {
		t.Fatal(err)
	}
	if a[0].Choice != "K1" || a[1].Choice != "K2" {
		t.Fatalf("examples ignored: %v / %v", a[0].Probs, a[1].Probs)
	}
	if a[0].Score <= 0 || a[0].Score > 1 {
		t.Fatalf("cosine %v", a[0].Score)
	}
}

func TestEmbedCacheAndMargin(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1, WithEmbedCache(2))
	if err != nil {
		t.Fatal(err)
	}
	a, _ := m.Embed(ctx, "un", "deux")
	b, _ := m.Embed(ctx, "deux", "trois") // "un" falls out of the cache
	if &a[1][0] != &b[0][0] {
		t.Fatal("\"deux\" should have come from the cache")
	}
	if _, ok := m.embedCache.get("un"); ok {
		t.Fatal("the cache exceeds its size")
	}
	ans, _ := m.ChooseNearest(ctx, []Candidate{{Name: "facture"}, {Name: "panne"}, {Name: "réunion"}}, "Ma facture est fausse")
	p := ans[0].Probs
	var others float64
	for n, v := range p {
		if n != ans[0].Choice {
			others += v
		}
	}
	if math.Abs(ans[0].Margin-(ans[0].Confidence-others/2)) > 1e-12 {
		t.Fatalf("margin %v", ans[0].Margin)
	}
}

func TestDecideOpen(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1, WithEmbedCache(64))
	if err != nil {
		t.Fatal(err)
	}
	qs := []OpenQuestion{
		{Name: "facture", Kind: Noul, Options: []Candidate{
			{Name: "oui", Description: "Le courriel parle d'une facture ou d'un paiement."},
			{Name: "non", Description: "Le courriel parle d'autre chose que de factures."},
		}},
		{Name: "urgence", Kind: Score, Instructions: "Urgence du courriel", Options: []Candidate{{Name: "faible"}, {Name: "moyenne"}, {Name: "haute"}}},
		{Name: "service", Kind: Choice, Options: []Candidate{{Name: "comptabilité"}, {Name: "informatique"}}},
	}
	d, err := m.DecideOpen(ctx, qs, "Votre facture de 1 200 € est impayée", "Le serveur est tombé en panne")
	if err != nil {
		t.Fatal(err)
	}
	if d[0]["facture"].P <= d[1]["facture"].P {
		t.Errorf("noul: %v then %v", d[0]["facture"].P, d[1]["facture"].P)
	}
	if s := d[0]["urgence"].Score; s < 0 || s > 2 {
		t.Errorf("score out of scale: %v", s)
	}
	if d[0]["service"].Choice != "comptabilité" || d[1]["service"].Choice != "informatique" {
		t.Errorf("choice: %v / %v", d[0]["service"].Probs, d[1]["service"].Probs)
	}
	if _, err := m.DecideOpen(ctx, []OpenQuestion{{Name: "x", Kind: Noul, Options: []Candidate{{Name: "a"}}}}, "t"); err == nil {
		t.Error("noul with one criterion accepted")
	}
}

// FitEmbeddings with a teacher: with both weights at zero, the teacher is
// not run and the model trains exactly as without it; with a weight, the
// training runs to its end and every distinct text is cached once.
func TestFitEmbeddingsTeacher(t *testing.T) {
	newModel := func() *Model {
		m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	b := []ChoiceBatch{{
		Texts:      []string{"Ma facture de mars est fausse", "Le serveur ne répond plus"},
		Candidates: []Candidate{{Name: "Facturation"}, {Name: "Support technique"}},
		Correct:    [][]int{{0}, {1}},
	}}
	opts := DefaultTrainOptions()
	opts.Epochs, opts.LR = 2, 1e-4
	plain, idle := newModel(), newModel()
	if err := plain.FitEmbeddings(context.Background(), b, opts); err != nil {
		t.Fatal(err)
	}
	orphan := opts
	orphan.DistillHidden = 1
	if err := newModel().FitEmbeddings(context.Background(), b, orphan); err == nil {
		t.Fatal("DistillHidden without a Teacher was accepted")
	}
	cold := opts
	cold.Teacher, cold.DistillHidden, cold.DistillTemp = newModel(), 1, -1
	if err := newModel().FitEmbeddings(context.Background(), b, cold); err == nil {
		t.Fatal("a negative DistillTemp was accepted")
	}
	withIdle := opts
	withIdle.Teacher = newModel()
	if err := idle.FitEmbeddings(context.Background(), b, withIdle); err != nil {
		t.Fatal(err)
	}
	for i, w := range plain.enc.FinalNorm.W {
		if idle.enc.FinalNorm.W[i] != w {
			t.Fatalf("gamma[%d]: %g with an idle teacher, %g without", i, idle.enc.FinalNorm.W[i], w)
		}
	}

	// The teacher's questions play no part: one with another schema teaches.
	other, err := New(bekkoDir(t), Schema{NewNoul("other", "")}, 1)
	if err != nil {
		t.Fatal(err)
	}
	active := opts
	active.Teacher, active.DistillHidden, active.Distill = other, 1, 0.5
	var last float64
	active.Progress = func(p Progress) { last = p.Loss }
	if err := newModel().FitEmbeddings(context.Background(), b, active); err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(last) || math.IsInf(last, 0) {
		t.Fatalf("loss %g", last)
	}
	tc := newEmbedTeacher(active.Teacher, 0.5, 1, 1)
	m := newModel()
	for range 2 {
		if _, _, err := m.choiceStep(b[0], DefaultEmbedScale, false, tc, m.enc.EnableGrad()); err != nil {
			t.Fatal(err)
		}
	}
	if n := tc.units.order.Len(); n != 4 {
		t.Fatalf("%d texts cached, want 4 (2 texts, 2 options)", n)
	}
}
