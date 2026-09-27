package main

import (
	"context"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/dataset/synth"
)

// mixOptions choisit les sources d'entraînement en plus des tickets et des
// courriels Enron étiquetés.
type mixOptions struct {
	fr       bool   // courriels Enron traduits en français
	synth    int    // courriels synthétiques en français (0 : aucun)
	synthDir string // gabarits
}

var mix mixOptions

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

// listBatches regroupe les courriels étiquetés selon les listes
// d'entraînement : pour chaque liste, des lots de courriels dont les options
// sont toutes les catégories de la liste.
func listBatches(emails []dataset.Example, lists []taxonomy, size int, rng *rand.Rand) []indecis.ChoiceBatch {
	var out []indecis.ChoiceBatch
	for _, t := range lists {
		index := map[string]int{}
		for i, o := range t.Options {
			index[o.Name] = i
		}
		type item struct {
			text string
			pos  int
		}
		var items []item
		for _, e := range emails {
			if v, ok := e.Labels[t.Name]; ok {
				if j, ok := index[topLabel(v)]; ok {
					items = append(items, item{e.Text, j})
				}
			}
		}
		rng.Shuffle(len(items), func(i, j int) { items[i], items[j] = items[j], items[i] })
		for s := 0; s+size <= len(items); s += size {
			b := indecis.ChoiceBatch{Candidates: t.Options}
			for _, it := range items[s : s+size] {
				b.Texts = append(b.Texts, it.text)
				b.Correct = append(b.Correct, []int{it.pos})
			}
			out = append(out, b)
		}
	}
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
	sources := []struct {
		file  string
		lists []taxonomy
		use   bool
	}{
		{"enron_train.jsonl", training, true},
		{"enron_train_fr.jsonl", training, mix.fr},
	}
	for _, src := range sources {
		if !src.use {
			continue
		}
		if emails, err := dataset.ReadFile(filepath.Join(dir, src.file)); err == nil {
			eb := listBatches(emails, src.lists, 16, rng)
			log.Printf("%s : %d courriels → %d lots", src.file, len(emails), len(eb))
			batches = append(batches, eb...)
		}
	}
	if mix.synth > 0 {
		c, err := synth.LoadFS(os.DirFS(mix.synthDir), synth.DefaultGazetteerOptions())
		if err != nil {
			return err
		}
		gen, err := c.Generate(mix.synth, synth.Options{Seed: 1, Dedupe: true})
		if err != nil {
			return err
		}
		eb := listBatches(gen, templated, 16, rng)
		log.Printf("gabarits : %d courriels → %d lots", len(gen), len(eb))
		batches = append(batches, eb...)
	}
	rng.Shuffle(len(batches), func(i, j int) { batches[i], batches[j] = batches[j], batches[i] })

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
			if err := report(ctx, x.label, nearest(x.m), s); err != nil {
				return err
			}
		}
	}
	return nil
}

func nearest(m *indecis.Model) chooser {
	return func(ctx context.Context, cands []indecis.Candidate, texts []string) ([]indecis.Answer, error) {
		return m.ChooseNearest(ctx, cands, texts...)
	}
}
