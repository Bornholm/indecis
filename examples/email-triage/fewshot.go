package main

import (
	"context"
	"fmt"
	"math"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// fewShot mesure le classement quand chaque catégorie est définie, en plus
// de son nom, par k courriels d'exemple : le prototype de la catégorie est
// la moyenne du plongement du nom et de ceux des exemples. Les exemples sont
// retirés du jeu de test. La liste reste modifiable à l'inférence : ajouter
// une catégorie, c'est donner son nom et quelques courriels.
func fewShot(ctx context.Context, label string, m *indecis.Model, s evalSet, k int) error {
	byCat := map[string][]int{}
	for i, e := range s.ex {
		g := s.gold(e)
		if len(g) == 1 {
			byCat[g[0]] = append(byCat[g[0]], i)
		}
	}
	shots := map[int]bool{}
	protoTexts := map[string][]string{}
	for _, c := range s.cands {
		idx := byCat[c.Name]
		for _, i := range idx[:min(k, len(idx))] {
			shots[i] = true
			protoTexts[c.Name] = append(protoTexts[c.Name], s.ex[i].Text)
		}
	}
	var test []dataset.Example
	for i, e := range s.ex {
		if !shots[i] {
			test = append(test, e)
		}
	}
	protos := make([][]float64, len(s.cands))
	for j, c := range s.cands {
		vecs, err := m.Embed(ctx, append([]string{indecis.CandidateContext(c)}, protoTexts[c.Name]...)...)
		if err != nil {
			return err
		}
		p := make([]float64, len(vecs[0]))
		for _, v := range vecs {
			for d, x := range v {
				p[d] += float64(x)
			}
		}
		var n float64
		for _, x := range p {
			n += x * x
		}
		for d := range p {
			p[d] /= math.Sqrt(n)
		}
		protos[j] = p
	}
	texts := make([]string, len(test))
	for i, e := range test {
		texts[i] = e.Text
	}
	te, err := m.Embed(ctx, texts...)
	if err != nil {
		return err
	}
	correct := 0
	for i, t := range te {
		best, bs := "", math.Inf(-1)
		for j, p := range protos {
			var dot float64
			for d, x := range t {
				dot += float64(x) * p[d]
			}
			if dot > bs {
				best, bs = s.cands[j].Name, dot
			}
		}
		if contains(s.gold(test[i]), best) {
			correct++
		}
	}
	fmt.Printf("%-10s %-38s n=%-5d catégories=%-3d exactitude=%5.1f%%  (nom + %d exemples par catégorie)\n",
		label, s.name, len(test), len(s.cands), 100*float64(correct)/float64(len(test)), k)
	return nil
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
	tuned, err := indecis.Load(model, indecis.WithInt8())
	if err != nil {
		return err
	}
	for _, s := range evalSets(tickets, imnim, enron) {
		for _, x := range []struct {
			label string
			m     *indecis.Model
		}{{"backbone", base}, {"affiné", tuned}} {
			if err := fewShot(ctx, x.label, x.m, s, k); err != nil {
				return err
			}
		}
	}
	return nil
}
