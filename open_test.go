package indecis

import (
	"context"
	"math"
	"testing"
)

// Reuses the example from bekko's model card: the query retrieves the
// document about sushi, and the embeddings are normalized.
func TestEmbedMatchesModelCard(t *testing.T) {
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1, WithPairs())
	if err != nil {
		t.Fatal(err)
	}
	docs := []string{
		"A warm noodle soup served in broth with sliced toppings.",
		"天ぷらは魚や野菜に衣をつけて揚げた料理です。",
		"Une fine crepe garnie de sucre, de beurre ou de fruits.",
		"A Japanese dish made with vinegared rice, often shaped with seafood, vegetables, or egg.",
	}
	e, err := m.Embed(context.Background(), append([]string{"What are the characteristics of sushi?"}, docs...)...)
	if err != nil {
		t.Fatal(err)
	}
	// Scores from the model card (sentence-transformers): 0.3085, 0.2716, 0.2750, 0.4738.
	want := []float64{0.3085, 0.2716, 0.2750, 0.4738}
	for i := range docs {
		var dot, n float64
		for j := range e[0] {
			dot += float64(e[0][j]) * float64(e[i+1][j])
			n += float64(e[i+1][j]) * float64(e[i+1][j])
		}
		if math.Abs(n-1) > 1e-5 {
			t.Fatalf("norm %v", n)
		}
		if math.Abs(dot-want[i]) > 2e-3 {
			t.Errorf("document %d: cosine %.4f, expected %.4f", i, dot, want[i])
		}
	}
}

func TestChooseAmong(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1, WithPairs())
	if err != nil {
		t.Fatal(err)
	}
	cands := []Candidate{{Name: "facturation", Description: "factures, paiements"}, {Name: "support"}, {Name: "rh"}}
	a, err := m.ChooseAmong(ctx, "match", cands, "Ma facture est fausse", "Mon ordinateur ne démarre plus")
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range a {
		var sum float64
		for _, p := range x.Probs {
			sum += p
		}
		if len(x.Probs) != 3 || math.Abs(sum-1) > 1e-9 || x.Probs[x.Choice] != x.Confidence || x.P <= 0 || x.P >= 1 {
			t.Fatalf("inconsistent answer: %+v", x)
		}
	}
	if _, err := m.ChooseAmong(ctx, "match", []Candidate{{Name: "a"}, {Name: "a"}}, "x"); err == nil {
		t.Fatal("duplicate options accepted")
	}
	unpaired, _ := New(bekkoDir(t), Schema{NewNoul("match", "")}, 1)
	if _, err := unpaired.ChooseAmong(ctx, "match", cands, "x"); err == nil {
		t.Fatal("model without pairs accepted")
	}
}
