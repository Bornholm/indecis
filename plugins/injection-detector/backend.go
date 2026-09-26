package main

import (
	"context"
	"fmt"

	"github.com/bornholm/indecis"

	"github.com/bornholm/indecis/plugins/injection-detector/internal/detector"
)

// modelBackend adapte un modèle indecis à detector.Backend : la question
// noul donne le logit, la première question choice, s'il y en a une, la
// catégorie.
type modelBackend struct {
	m        *indecis.Model
	question string
	category string
	options  []string
	info     detector.ModelInfo
}

func newModelBackend(m *indecis.Model, question string) (*modelBackend, error) {
	b := &modelBackend{m: m, question: question}
	found := false
	for _, q := range m.Schema() {
		switch {
		case q.Name == question && q.Kind == indecis.Noul:
			found = true
		case q.Kind == indecis.Choice && b.category == "":
			b.category, b.options = q.Name, q.Options
		}
	}
	if !found {
		return nil, fmt.Errorf("injection-detector : le modèle n'a pas de question noul %q", question)
	}
	prior, ok := m.Info().TrainPrior[question]
	if !ok {
		prior = 0.5
	}
	b.info = detector.ModelInfo{
		Version:     fmt.Sprintf("%s/%d", m.Info().Backbone, m.Info().Steps),
		TrainPrior:  prior,
		Temperature: m.Temperatures()[question],
	}
	return b, nil
}

func (b *modelBackend) Info() detector.ModelInfo { return b.info }

// Fenêtres des segments longs, en caractères : le modèle ne lit qu'environ
// 256 tokens par entrée, une injection en fin de page lui échapperait.
const (
	windowRunes  = 800
	overlapRunes = 150
)

// windows découpe un texte long en fenêtres qui se chevauchent, coupées sur
// une espace quand c'est possible.
func windows(text string) []string {
	r := []rune(text)
	if len(r) <= windowRunes {
		return []string{text}
	}
	var out []string
	for start := 0; start < len(r); {
		end := min(start+windowRunes, len(r))
		if end < len(r) {
			for cut := end; cut > start+windowRunes/2; cut-- {
				if r[cut] == ' ' || r[cut] == '\n' {
					end = cut
					break
				}
			}
		}
		out = append(out, string(r[start:end]))
		if end == len(r) {
			break
		}
		start = max(end-overlapRunes, start+1)
	}
	return out
}

func (b *modelBackend) Score(ctx context.Context, segments []detector.Segment) ([]detector.Score, error) {
	var inputs []indecis.Input
	owner := []int{}
	for i, s := range segments {
		for _, w := range windows(s.Text) {
			in := indecis.Input{Text: w}
			// Un modèle sans paires ne sait pas lire le contexte.
			if b.m.Paired() {
				in.Context = s.Context
			}
			inputs = append(inputs, in)
			owner = append(owner, i)
		}
	}
	logits, err := b.m.LogitsInputs(ctx, inputs...)
	if err != nil {
		return nil, err
	}
	// Un segment vaut sa fenêtre la plus suspecte.
	out := make([]detector.Score, len(segments))
	seen := make([]bool, len(segments))
	for j, l := range logits {
		i := owner[j]
		z := l[b.question][0]
		if seen[i] && z <= out[i].Logit {
			continue
		}
		seen[i] = true
		out[i].Logit = z
		if b.category != "" {
			c := l[b.category]
			best := 0
			for k := range c {
				if c[k] > c[best] {
					best = k
				}
			}
			out[i].Category = b.options[best]
		}
	}
	return out, nil
}
