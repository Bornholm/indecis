package indecis

import (
	"context"
	"fmt"
	"math"

	"github.com/bornholm/indecis/dataset"
)

// Choice among options given at inference time.
//
// A Choice question has fixed options: they are outputs of its head,
// learned during training. For a list of categories to be able to change
// from one call to the next, the model must read the option instead of
// knowing it: a paired model answers a Noul question ("does the text fall
// under this option?") for each (option, text) pair, and ChooseAmong
// compares the answers. CandidatePairs produces the corresponding training
// examples.

// Candidate is an option described in natural language. Examples are texts
// that fall under the option: ChooseNearest uses them to locate the option
// (a few examples are often worth more than a long description);
// ChooseAmong ignores them.
type Candidate struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Examples    []string `json:"examples,omitempty"`
}

// CandidateContext puts an option in the form the model reads as context,
// both in training and inference.
func CandidateContext(c Candidate) string {
	if c.Description == "" {
		return c.Name
	}
	return c.Name + "\n" + c.Description
}

// CandidatePairs turns a text whose correct option is correct into
// examples for the Noul question named question: one positive, and one
// negative per other option.
func CandidatePairs(question, text string, correct Candidate, others []Candidate) []dataset.Example {
	out := []dataset.Example{{Context: CandidateContext(correct), Text: text, Labels: map[string]any{question: true}}}
	for _, o := range others {
		out = append(out, dataset.Example{Context: CandidateContext(o), Text: text, Labels: map[string]any{question: false}})
	}
	return out
}

// ChooseAmong picks, for each text, the option that suits it best among
// candidates, using the paired model's Noul question named question.
//
// Answer.Probs is the distribution over the options (softmax of the
// logits, including the question's temperature) and Answer.Choice the most
// probable one. Answer.P is the probability that this best option really
// suits the text: a low value signals a text that no option describes.
func (m *Model) ChooseAmong(ctx context.Context, question string, candidates []Candidate, texts ...string) ([]Answer, error) {
	if !m.paired {
		return nil, fmt.Errorf("indecis: ChooseAmong requires a paired model (WithPairs)")
	}
	qi := -1
	for i, q := range m.schema {
		if q.Name == question && q.Kind == Noul {
			qi = i
		}
	}
	if qi < 0 {
		return nil, fmt.Errorf("indecis: no noul question %q", question)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("indecis: no options")
	}
	seen := map[string]bool{}
	inputs := make([]Input, 0, len(texts)*len(candidates))
	for _, c := range candidates {
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("indecis: option %q empty or duplicate", c.Name)
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
		a.Margin = margin(p, best)
		a.P = 1 / (1 + math.Exp(-zs[best]))
		out[ti] = a
	}
	return out, nil
}

// Embed returns the embedding of each text: the mean of the encoder's
// states, normalized (norm 1). The dot product of two embeddings is their
// cosine similarity. Each text is encoded alone, even by a paired model: on
// the non-fine-tuned backbone, it is the original model's sentence
// embedding.
func (m *Model) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	return m.embed(ctx, texts, true)
}

// embed computes the embeddings; cache says whether the cache
// (WithEmbedCache) keeps them. A list's options recur from one call to the
// next, the texts being judged almost never: ChooseIn does not cache the
// latter.
func (m *Model) embed(ctx context.Context, texts []string, cache bool) ([][]float32, error) {
	out := make([][]float32, len(texts))
	var todo []int
	var ids [][]int32
	for i, t := range texts {
		if v, ok := m.embedCache.get(t); ok {
			out[i] = v
			continue
		}
		todo = append(todo, i)
		ids = append(ids, m.tok.EncodeMax(t, m.maxLen))
	}
	err := m.forEachPooled(ctx, ids, 32, func(k int, x []float32) {
		var n float64
		for _, v := range x {
			n += float64(v) * float64(v)
		}
		v := make([]float32, len(x))
		inv := float32(1 / math.Sqrt(max(n, 1e-24)))
		for j, e := range x {
			v[j] = e * inv
		}
		out[todo[k]] = v
		if cache {
			m.embedCache.put(texts[todo[k]], v)
		}
	})
	return out, err
}
