package main

import (
	"context"
	"fmt"

	"github.com/bornholm/indecis"
)

// fewShot mesure le classement quand chaque catégorie est définie, en plus
// de son nom, par k courriels d'exemple : le prototype de la catégorie est
// la moyenne du plongement du nom et de ceux des exemples. Les exemples sont
// retirés du jeu de test. La liste reste modifiable à l'inférence : ajouter
// une catégorie, c'est donner son nom et quelques courriels.
func fewShot(ctx context.Context, label string, m *indecis.Model, s evalSet, k int) error {
	byCat := map[string][]int{}
	for i, e := range s.ex {
		if g := s.gold(e); len(g) == 1 {
			byCat[g[0]] = append(byCat[g[0]], i)
		}
	}
	shots := map[int]bool{}
	cands := make([]indecis.Candidate, len(s.cands))
	for j, c := range s.cands {
		c.Examples = nil
		idx := byCat[c.Name]
		for _, i := range idx[:min(k, len(idx))] {
			shots[i] = true
			c.Examples = append(c.Examples, s.ex[i].Text)
		}
		cands[j] = c
	}
	test := evalSet{name: fmt.Sprintf("%s, + %d exemples", s.name, k), cands: cands, gold: s.gold}
	for i, e := range s.ex {
		if !shots[i] {
			test.ex = append(test.ex, e)
		}
	}
	return report(ctx, label, nearest(m), test)
}

func evaluateFewShot(ctx context.Context, dir, model, backbone string, k int) error {
	tickets, imnim, enron, err := load(dir)
	if err != nil {
		return err
	}
	base, err := indecis.New(backbone, indecis.Schema{indecis.NewNoul(question, "")}, 1, indecis.WithInt8())
	if err != nil {
		return err
	}
	models := []struct {
		label string
		m     *indecis.Model
	}{{"backbone", base}}
	if model != "-" { // "-" : backbone seul
		tuned, err := indecis.Load(model, indecis.WithInt8())
		if err != nil {
			return err
		}
		models = append(models, struct {
			label string
			m     *indecis.Model
		}{"affiné", tuned})
	}
	for _, s := range evalSets(tickets, imnim, enron) {
		for _, x := range models {
			if err := fewShot(ctx, x.label, x.m, s, k); err != nil {
				return err
			}
		}
	}
	return nil
}
