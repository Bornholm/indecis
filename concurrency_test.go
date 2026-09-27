package indecis

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// Des requêtes simultanées sur un même modèle donnent les mêmes réponses
// qu'une à une : à lancer avec -race.
func TestConcurrentInference(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), Schema{NewNoul("match", ""), NewChoice("c", "", "a", "b")}, 1, WithInt8(), WithEmbedCache(16))
	if err != nil {
		t.Fatal(err)
	}
	texts := make([]string, 24)
	for i := range texts {
		texts[i] = fmt.Sprintf("Courriel numéro %d : la facture %d est impayée depuis %d jours.", i, 1000+i, i%30)
	}
	cands := []Candidate{{Name: "facturation", Examples: []string{"Relance de paiement"}}, {Name: "informatique"}}
	want, _ := m.Decide(ctx, texts...)
	wantOpen, _ := m.ChooseNearest(ctx, cands, texts...)
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for r := 0; r < 3; r++ {
				i := (g*3 + r) % len(texts)
				d, err := m.Decide(ctx, texts[i])
				if err != nil {
					errs <- err
					return
				}
				if d[0]["match"].P != want[i]["match"].P {
					errs <- fmt.Errorf("Decide %d : %v ≠ %v", i, d[0]["match"].P, want[i]["match"].P)
				}
				a, err := m.ChooseNearest(ctx, cands, texts[i])
				if err != nil {
					errs <- err
					return
				}
				if a[0].Confidence != wantOpen[i].Confidence {
					errs <- fmt.Errorf("ChooseNearest %d : %v ≠ %v", i, a[0].Confidence, wantOpen[i].Confidence)
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
