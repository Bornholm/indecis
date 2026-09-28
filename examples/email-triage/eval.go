package main

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// evalSet is a test set: emails, the list of categories offered to
// the model, and the correct categories for each email (often only
// one).
type evalSet struct {
	name  string
	ex    []dataset.Example
	cands []indecis.Candidate
	gold  func(dataset.Example) []string
}

// chooser classifies texts among categories.
type chooser func(ctx context.Context, cands []indecis.Candidate, texts []string) ([]indecis.Answer, error)

// report measures accuracy (the chosen category is correct) and the
// macro F1, category by category.
func report(ctx context.Context, name string, choose chooser, s evalSet) error {
	texts := make([]string, len(s.ex))
	for i, e := range s.ex {
		texts[i] = e.Text
	}
	start := time.Now()
	answers, err := choose(ctx, s.cands, texts)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	tp, fp, fn := map[string]int{}, map[string]int{}, map[string]int{}
	correct, top3 := 0, 0
	for i, a := range answers {
		gold := s.gold(s.ex[i])
		ok := contains(gold, a.Choice)
		if ok {
			correct++
			tp[a.Choice]++
		} else {
			fp[a.Choice]++
			fn[gold[0]]++
		}
		for _, c := range topK(a.Probs, 3) {
			if contains(gold, c) {
				top3++
				break
			}
		}
	}
	var f1 float64
	for _, c := range s.cands {
		p := float64(tp[c.Name]) / math.Max(1, float64(tp[c.Name]+fp[c.Name]))
		r := float64(tp[c.Name]) / math.Max(1, float64(tp[c.Name]+fn[c.Name]))
		if p+r > 0 {
			f1 += 2 * p * r / (p + r)
		}
	}
	f1 /= float64(len(s.cands))
	n := float64(len(answers))
	fmt.Printf("%-10s %-38s n=%-5d categories=%-3d acc=%5.1f%%  top-3=%5.1f%%  macroF1=%.3f  %5.1f ms/email\n",
		name, s.name, len(answers), len(s.cands), 100*float64(correct)/n, 100*float64(top3)/n, f1,
		float64(elapsed.Microseconds())/1000/n)
	return nil
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func topK(p map[string]float64, k int) []string {
	names := make([]string, 0, len(p))
	for n := range p {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return p[names[i]] > p[names[j]] })
	return names[:min(k, len(names))]
}

// embedChooser compares the embeddings of the email and the
// categories: the training-free method, with the backbone as is.
func embedChooser(m *indecis.Model, temperature float64) chooser {
	return func(ctx context.Context, cands []indecis.Candidate, texts []string) ([]indecis.Answer, error) {
		ctxs := make([]string, len(cands))
		for i, c := range cands {
			ctxs[i] = indecis.CandidateContext(c)
		}
		ce, err := m.Embed(ctx, ctxs...)
		if err != nil {
			return nil, err
		}
		te, err := m.Embed(ctx, texts...)
		if err != nil {
			return nil, err
		}
		out := make([]indecis.Answer, len(texts))
		for i, t := range te {
			z := make([]float64, len(cands))
			mx := math.Inf(-1)
			for j, c := range ce {
				var dot float64
				for k := range t {
					dot += float64(t[k]) * float64(c[k])
				}
				z[j] = dot / temperature
				mx = max(mx, z[j])
			}
			var sum float64
			for j := range z {
				z[j] = math.Exp(z[j] - mx)
				sum += z[j]
			}
			a := indecis.Answer{Kind: indecis.Choice, Probs: map[string]float64{}}
			for j, c := range cands {
				p := z[j] / sum
				a.Probs[c.Name] = p
				if p > a.Confidence {
					a.Confidence, a.Choice = p, c.Name
				}
			}
			out[i] = a
		}
		return out, nil
	}
}

func pairChooser(m *indecis.Model) chooser {
	return func(ctx context.Context, cands []indecis.Candidate, texts []string) ([]indecis.Answer, error) {
		return m.ChooseAmong(ctx, question, cands, texts...)
	}
}

const question = "match"

// Ticket split. Part of the queues is kept entirely out of training:
// their tickets measure classification toward categories unknown to
// the model, within a known domain.
func heldOutQueue(q string) bool { return hash(q)%4 == 0 }

func testTicket(e dataset.Example) bool { return hash(e.Text)%10 == 0 }

func hash(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// dataDir is the data directory, for optional test sets.
var dataDir string

// evalSets builds the test sets from the prepared data.
func evalSets(tickets, imnim, enron []dataset.Example) []evalSet {
	var sets []evalSet
	queues := map[string]bool{}
	for _, e := range tickets {
		queues[e.Meta["queue"]] = true
	}
	var allQueues []indecis.Candidate
	for q := range queues {
		allQueues = append(allQueues, indecis.Candidate{Name: q})
	}
	sort.Slice(allQueues, func(i, j int) bool { return allQueues[i].Name < allQueues[j].Name })
	var seen, unseen []dataset.Example
	for _, e := range tickets {
		switch {
		case heldOutQueue(e.Meta["queue"]):
			unseen = append(unseen, e)
		case testTicket(e):
			seen = append(seen, e)
		}
	}
	queueGold := func(e dataset.Example) []string { return []string{e.Meta["queue"]} }
	sets = append(sets,
		evalSet{"tickets, seen queues (new tickets)", cap500(seen), allQueues, queueGold},
		evalSet{"tickets, never-seen queues", cap500(unseen), allQueues, queueGold})

	// imnim: emails with a single category, among its 10 categories.
	var single []dataset.Example
	labels := map[string]bool{}
	for _, e := range imnim {
		ls := strings.Split(e.Meta["labels"], "|")
		for _, l := range ls {
			labels[l] = true
		}
		if len(ls) == 1 {
			single = append(single, e)
		}
	}
	var imnimCands []indecis.Candidate
	for l := range labels {
		imnimCands = append(imnimCands, indecis.Candidate{Name: l})
	}
	sort.Slice(imnimCands, func(i, j int) bool { return imnimCands[i].Name < imnimCands[j].Name })
	sets = append(sets, evalSet{"imnim (never-seen categories)", cap500(single), imnimCands,
		func(e dataset.Example) []string { return strings.Split(e.Meta["labels"], "|") }})

	// ASN: real letters in French, inspection theme to identify.
	if asn, err := dataset.ReadFile(filepath.Join(dataDir, "asn.jsonl")); err == nil && len(asn) > 0 {
		themes := map[string]bool{}
		for _, e := range asn {
			themes[e.Meta["theme"]] = true
		}
		var cands []indecis.Candidate
		for t := range themes {
			cands = append(cands, indecis.Candidate{Name: t})
		}
		sort.Slice(cands, func(i, j int) bool { return cands[i].Name < cands[j].Name })
		rng := rand.New(rand.NewSource(5))
		rng.Shuffle(len(asn), func(i, j int) { asn[i], asn[j] = asn[j], asn[i] })
		sets = append(sets, evalSet{"ASN, FR letters, theme", cap500(asn), cands,
			func(e dataset.Example) []string { return []string{e.Meta["theme"]} }})
	}
	// Enron translated to French, same list and same labels.
	if fr, err := dataset.ReadFile(filepath.Join(dataDir, "enron_labeled_fr.jsonl")); err == nil && len(fr) > 0 {
		sets = append(sets, evalSet{"Enron translated (FR), classic list", fr, classic,
			func(e dataset.Example) []string { return []string{topLabel(e.Labels["categorie"])} }})
	}

	// Enron, labeled by the teachers according to the classic list.
	var labeled []dataset.Example
	for _, e := range enron {
		if _, ok := e.Labels["categorie"]; ok {
			labeled = append(labeled, e)
		}
	}
	if len(labeled) > 0 {
		sets = append(sets, evalSet{"Enron, classic list (FR)", labeled, classic,
			func(e dataset.Example) []string { return []string{topLabel(e.Labels["categorie"])} }})
	}
	return sets
}

// topLabel reads a choice label: an option, or a distribution.
func topLabel(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case map[string]any:
		best, bp := "", -1.0
		for k, p := range x {
			if f, _ := p.(float64); f > bp {
				best, bp = k, f
			}
		}
		return best
	}
	return ""
}

func cap500(ex []dataset.Example) []dataset.Example {
	if len(ex) > 500 {
		return ex[:500]
	}
	return ex
}
