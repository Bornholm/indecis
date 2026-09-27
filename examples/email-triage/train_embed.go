package main

import (
	"context"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// choiceBatches regroupe des tickets par lots d'une même liste (files,
// types, mots-clés). Les options d'un lot sont les catégories justes de ses
// tickets, complétées par des catégories tirées selon leur fréquence, pour
// que le nom d'une catégorie ne dise rien de sa justesse.
func choiceBatches(tickets []dataset.Example, size, options int, rng *rand.Rand) []indecis.ChoiceBatch {
	type item struct {
		text string
		pos  string
		all  map[string]bool
	}
	lists := map[string][]item{}
	for _, e := range tickets {
		all := map[string]bool{e.Meta["queue"]: true, e.Meta["type"]: true}
		tags := strings.Split(e.Meta["tags"], "|")
		for _, t := range tags {
			all[t] = true
		}
		lists["queue"] = append(lists["queue"], item{e.Text, e.Meta["queue"], all})
		if t := e.Meta["type"]; t != "" {
			lists["type"] = append(lists["type"], item{e.Text, t, all})
		}
		if tags[0] != "" {
			lists["tag"] = append(lists["tag"], item{e.Text, tags[rng.Intn(min(3, len(tags)))], all})
		}
	}
	var out []indecis.ChoiceBatch
	for _, name := range []string{"queue", "type", "tag"} {
		items := lists[name]
		rng.Shuffle(len(items), func(i, j int) { items[i], items[j] = items[j], items[i] })
		distinct := map[string]bool{}
		for _, it := range items {
			distinct[it.pos] = true
		}
		for s := 0; s+size <= len(items); s += size {
			group := items[s : s+size]
			var cands []indecis.Candidate
			index := map[string]int{}
			addCand := func(n string) {
				if _, ok := index[n]; !ok {
					index[n] = len(cands)
					cands = append(cands, candidate(n, rng))
				}
			}
			for _, it := range group {
				addCand(it.pos)
			}
			for tries := 0; len(cands) < min(options, len(distinct)) && tries < 10*options; tries++ {
				addCand(items[rng.Intn(len(items))].pos)
			}
			b := indecis.ChoiceBatch{Candidates: cands}
			for _, it := range group {
				correct := []int{index[it.pos]}
				for n, j := range index {
					if n != it.pos && it.all[n] {
						correct = append(correct, j) // juste aussi : écartée
					}
				}
				b.Texts = append(b.Texts, it.text)
				b.Correct = append(b.Correct, correct)
			}
			out = append(out, b)
		}
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

func trainEmbed(ctx context.Context, dir, backbone, out string, n, epochs int) error {
	tickets, _, _, err := load(dir)
	if err != nil {
		return err
	}
	var pool []dataset.Example
	for _, e := range tickets {
		if !heldOutQueue(e.Meta["queue"]) && !testTicket(e) {
			pool = append(pool, e)
		}
	}
	rng := rand.New(rand.NewSource(1))
	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	pool = pool[:min(n, len(pool))]
	batches := choiceBatches(pool, 16, 24, rng)
	log.Printf("%d tickets → %d lots de 16", len(pool), len(batches))

	m, err := indecis.New(backbone, indecis.Schema{indecis.NewNoul(question, "")}, 1)
	if err != nil {
		return err
	}
	opts := indecis.DefaultTrainOptions()
	opts.Epochs, opts.LR = epochs, 2e-5
	opts.Progress = func(p indecis.Progress) {
		if p.Step%50 == 0 || p.Step == p.Steps {
			log.Printf("pas %d/%d perte %.4f — %.0f tokens/s, %s", p.Step, p.Steps, p.Loss,
				float64(p.Tokens)/p.Elapsed.Seconds(), p.Elapsed.Round(time.Second))
		}
	}
	if err := m.FitEmbeddings(ctx, batches, opts); err != nil {
		return err
	}
	if err := m.Save(out); err != nil {
		return err
	}
	log.Printf("modèle écrit dans %s", out)
	return evaluateEmbed(ctx, dir, out, backbone)
}

// evaluateEmbed compare les plongements du backbone et ceux du modèle affiné.
func evaluateEmbed(ctx context.Context, dir, model, backbone string) error {
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
		if err := report(ctx, "backbone", nearest(base), s); err != nil {
			return err
		}
		if err := report(ctx, "affiné", nearest(tuned), s); err != nil {
			return err
		}
	}
	return nil
}

func nearest(m *indecis.Model) chooser {
	return func(ctx context.Context, cands []indecis.Candidate, texts []string) ([]indecis.Answer, error) {
		return m.ChooseNearest(ctx, cands, texts...)
	}
}
