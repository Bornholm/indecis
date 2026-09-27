package indecis

import (
	"context"
	"fmt"
	"math"

	"github.com/bornholm/indecis/dataset"
)

// Choix parmi des options données à l'inférence.
//
// Une question Choice a des options fixes : ce sont des sorties de sa tête,
// apprises à l'entraînement. Pour qu'une liste de catégories puisse changer
// d'un appel à l'autre, le modèle doit lire l'option au lieu de la
// connaître : un modèle en paires répond à une question Noul (« le texte
// relève-t-il de cette option ? ») pour chaque paire (option, texte), et
// ChooseAmong compare les réponses. CandidatePairs produit les exemples
// d'entraînement correspondants.

// Candidate est une option décrite en langue naturelle.
type Candidate struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// CandidateContext met une option sous la forme que le modèle lit en
// contexte, à l'entraînement comme à l'inférence.
func CandidateContext(c Candidate) string {
	if c.Description == "" {
		return c.Name
	}
	return c.Name + "\n" + c.Description
}

// CandidatePairs transforme un texte dont l'option juste est correct en
// exemples pour la question Noul question : un positif, et un négatif par
// autre option.
func CandidatePairs(question, text string, correct Candidate, others []Candidate) []dataset.Example {
	out := []dataset.Example{{Context: CandidateContext(correct), Text: text, Labels: map[string]any{question: true}}}
	for _, o := range others {
		out = append(out, dataset.Example{Context: CandidateContext(o), Text: text, Labels: map[string]any{question: false}})
	}
	return out
}

// ChooseAmong choisit, pour chaque texte, l'option qui lui convient le mieux
// parmi candidates, par la question Noul question d'un modèle en paires.
//
// Answer.Probs est la distribution sur les options (softmax des logits,
// température de la question comprise) et Answer.Choice la plus probable.
// Answer.P est la probabilité que cette meilleure option convienne
// vraiment : une valeur basse signale un texte qu'aucune option ne décrit.
func (m *Model) ChooseAmong(ctx context.Context, question string, candidates []Candidate, texts ...string) ([]Answer, error) {
	if !m.paired {
		return nil, fmt.Errorf("indecis: ChooseAmong exige un modèle en paires (WithPairs)")
	}
	qi := -1
	for i, q := range m.schema {
		if q.Name == question && q.Kind == Noul {
			qi = i
		}
	}
	if qi < 0 {
		return nil, fmt.Errorf("indecis: pas de question noul %q", question)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("indecis: aucune option")
	}
	seen := map[string]bool{}
	inputs := make([]Input, 0, len(texts)*len(candidates))
	for _, c := range candidates {
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("indecis: option %q vide ou en double", c.Name)
		}
		seen[c.Name] = true
	}
	for _, t := range texts {
		for _, c := range candidates {
			inputs = append(inputs, Input{Context: CandidateContext(c), Text: t})
		}
	}
	ids, err := m.tokenizeAll(inputs)
	if err != nil {
		return nil, err
	}
	h, temp := m.heads[qi], m.temps[qi]
	z := make([]float64, len(inputs))
	err = m.forEachPooled(ctx, ids, 32, func(i int, x []float32) { z[i] = h.logits(x)[0] / temp })
	if err != nil {
		return nil, err
	}
	out := make([]Answer, len(texts))
	for ti := range texts {
		zs := z[ti*len(candidates) : (ti+1)*len(candidates)]
		p := softmax(zs)
		a := Answer{Question: question, Kind: Choice, Probs: make(map[string]float64, len(p))}
		best := 0
		for i, v := range p {
			a.Probs[candidates[i].Name] = v
			if v > p[best] {
				best = i
			}
		}
		a.Choice = candidates[best].Name
		a.Confidence = p[best]
		a.P = 1 / (1 + math.Exp(-zs[best]))
		out[ti] = a
	}
	return out, nil
}

// Embed retourne le plongement de chaque texte : moyenne des états de
// l'encodeur, normalisée (norme 1). Le produit scalaire de deux plongements
// est leur similarité cosinus. Chaque texte est encodé seul, même par un
// modèle en paires : sur le backbone non affiné, c'est l'embedding de phrase
// du modèle d'origine.
func (m *Model) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	ids := make([][]int32, len(texts))
	for i, t := range texts {
		ids[i] = m.tok.EncodeMax(t, m.maxLen)
	}
	out := make([][]float32, len(texts))
	err := m.forEachPooled(ctx, ids, 32, func(i int, x []float32) {
		var n float64
		for _, v := range x {
			n += float64(v) * float64(v)
		}
		v := make([]float32, len(x))
		inv := float32(1 / math.Sqrt(max(n, 1e-24)))
		for j, e := range x {
			v[j] = e * inv
		}
		out[i] = v
	})
	return out, err
}
