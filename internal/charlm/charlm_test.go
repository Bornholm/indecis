package charlm

import (
	"math"
	"path/filepath"
	"testing"
)

func TestModel(t *testing.T) {
	b := NewBuilder(4)
	for range 20 {
		b.Add("le chat noir dort sur le tapis")
		b.Add("la nuit tous les chats sont gris")
	}
	m := b.Build(1)
	// A distribution over the characters seen.
	var sum float64
	for _, c := range []rune(" abcdefghijklmnopqrstuvwxyz") {
		sum += m.Prob([]rune("le c"), c)
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("probabilities sum to %v", sum)
	}
	if p, q := m.Prob([]rune("le c"), 'h'), m.Prob([]rune("le c"), 'x'); p < 0.5 || q > 0.05 {
		t.Fatalf("P(h|le c) = %v, P(x|le c) = %v", p, q)
	}
	path := filepath.Join(t.TempDir(), "lm.bin")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	l, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if l.Prob([]rune("tapi"), 's') != m.Prob([]rune("tapi"), 's') || l.Order != 4 {
		t.Fatal("loaded model differs")
	}
}
