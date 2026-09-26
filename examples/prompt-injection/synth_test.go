package main

import (
	"os"
	"strings"
	"testing"

	"github.com/bornholm/indecis/dataset/synth"
)

// Le corpus de gabarits doit se charger, et ses étiquettes rester cohérentes
// avec le texte : une attaque connue de chaque famille ne doit jamais sortir
// étiquetée bénigne.
func TestSynthCorpus(t *testing.T) {
	c, err := synth.LoadFS(os.DirFS("synth"), synth.DefaultGazetteerOptions())
	if err != nil {
		t.Fatal(err)
	}
	ex, err := c.Generate(3000, synth.Options{Seed: 1, Dedupe: true})
	if err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, e := range ex {
		counts[e.Labels["category"].(string)]++
		inj := e.Labels["injection"].(bool)
		if inj != (e.Labels["category"] != "none") {
			t.Fatalf("étiquettes incohérentes : %v pour %q", e.Labels, e.Text)
		}
		if strings.HasPrefix(e.Family, "attack/") && !inj {
			t.Fatalf("attaque étiquetée bénigne : %q", e.Text)
		}
	}
	withCtx := 0
	for _, e := range ex {
		if e.Context != "" {
			withCtx++
		}
		// Un texte bénin générique ne reçoit jamais un prompt système
		// spécialisé : il deviendrait hors périmètre.
		if !strings.HasPrefix(e.Family, "attack/") && !strings.HasPrefix(e.Family, "scope/") && strings.Contains(e.Context, "Only help") {
			t.Fatalf("contexte spécialisé sur un texte bénin : %q", e.Context)
		}
	}
	t.Logf("catégories : %v, %d avec contexte", counts, withCtx)
	for i, e := range ex[:12] {
		t.Logf("%d [%s %v] %q", i, e.Family, e.Labels, e.Text)
	}
}
