package indecis

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/bornholm/indecis/internal/modernbert"
)

// Text embeddings for search: a query and the documents it should find
// are embedded on their own, then compared by their dot product (their
// cosine, the vectors having norm 1).

// EmbedQueries returns the embeddings of search queries, each preceded
// by the model's query prompt (Info.Prompts["query"]), as
// sentence-transformers does with prompt_name="query". Without a prompt,
// it is Embed.
func (m *Model) EmbedQueries(ctx context.Context, texts ...string) ([][]float32, error) {
	return m.embed(ctx, m.prompted("query", texts), false)
}

// EmbedDocuments returns the embeddings of the documents to search, each
// preceded by the model's document prompt (Info.Prompts["document"]).
// Without a prompt, it is Embed.
func (m *Model) EmbedDocuments(ctx context.Context, texts ...string) ([][]float32, error) {
	return m.embed(ctx, m.prompted("document", texts), false)
}

func (m *Model) prompted(role string, texts []string) []string {
	p := m.info.Prompts[role]
	if p == "" {
		return texts
	}
	out := make([]string, len(texts))
	for i, t := range texts {
		out[i] = p + t
	}
	return out
}

// Dimension is the size of the embeddings.
func (m *Model) Dimension() int { return m.enc.Cfg.Hidden }

// MaxLen is the number of tokens read at most per text (WithMaxLen).
func (m *Model) MaxLen() int { return m.maxLen }

// Truncate keeps the first dim components of an embedding and brings it
// back to norm 1: the Matryoshka truncation of models trained for it
// (granite-embedding R2 down to 128, Qwen3-Embedding down to 32). Other
// models lose much more quality. dim <= 0 or beyond the size returns v.
func Truncate(v []float32, dim int) []float32 {
	if dim <= 0 || dim >= len(v) {
		return v
	}
	var n float64
	for _, x := range v[:dim] {
		n += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(max(n, 1e-24)))
	out := make([]float32, dim)
	for i, x := range v[:dim] {
		out[i] = x * inv
	}
	return out
}

// readPrompts reads the query and document prompts of a
// sentence-transformers backbone (config_sentence_transformers.json).
// A prompt is a prefix of the text: with mean pooling, sentence-transformers
// may leave its tokens out of the mean (include_prompt false), which
// indecis does not reproduce, and refuses.
func readPrompts(dir string, cfg modernbert.Config) (map[string]string, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config_sentence_transformers.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var st struct {
		Prompts map[string]string `json:"prompts"`
	}
	if err := json.Unmarshal(b, &st); err != nil {
		return nil, fmt.Errorf("indecis: config_sentence_transformers.json: %w", err)
	}
	out := map[string]string{}
	for k, v := range st.Prompts {
		if v != "" {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil, nil
	}
	if !cfg.CLS() {
		pb, err := os.ReadFile(filepath.Join(dir, "1_Pooling", "config.json"))
		if err == nil {
			var p struct {
				IncludePrompt *bool `json:"include_prompt"`
			}
			if json.Unmarshal(pb, &p) == nil && p.IncludePrompt != nil && !*p.IncludePrompt {
				return nil, fmt.Errorf("indecis: %s: prompts left out of the mean (include_prompt false) are not supported", dir)
			}
		}
	}
	return out, nil
}

// SearchPair is a search query and a document it should find, for
// fine-tuning an embedding model (SearchBatches, then FitEmbeddings).
type SearchPair struct {
	// QueryID groups the pairs of one query (several relevant documents);
	// empty, the query text stands for it.
	QueryID string `json:"query_id,omitempty"`
	Query   string `json:"query"`
	// DocumentID identifies the document; empty, its text stands for it.
	DocumentID string `json:"document_id,omitempty"`
	Document   string `json:"document"`
	// Negatives are documents close to the query but not relevant (hard
	// negatives, mined by BM25 or by the model itself). Optional: the
	// other documents of the batch are negatives anyway.
	Negatives []string `json:"negatives,omitempty"`
}

// SearchBatches forms the FitEmbeddings batches of a search fine-tuning:
// in each batch, every query must prefer its document over the documents
// of the other queries and over the hard negatives, the
// MultipleNegativesRanking loss of sentence-transformers. Two pairs of
// the same query never share a batch, and a document relevant to a query
// other than as its target is left out of its comparison. The model's
// query and document prompts are put first, as EmbedQueries and
// EmbedDocuments do.
func (m *Model) SearchBatches(pairs []SearchPair, size int, seed int64) []ChoiceBatch {
	if size < 2 {
		size = 2
	}
	qid := func(p SearchPair) string { return cmp.Or(p.QueryID, p.Query) }
	did := func(p SearchPair) string { return cmp.Or(p.DocumentID, p.Document) }
	relevant := map[string]map[string]bool{}
	// A negative that is also a known document takes that document's id:
	// it may be relevant to another query of the batch.
	byText := map[string]string{}
	for _, p := range pairs {
		byText[p.Document] = did(p)
	}
	for _, p := range pairs {
		if relevant[qid(p)] == nil {
			relevant[qid(p)] = map[string]bool{}
		}
		relevant[qid(p)][did(p)] = true
	}
	rng := rand.New(rand.NewSource(seed))
	order := rng.Perm(len(pairs))
	// Greedy filling: a pair whose query is already in the group waits for
	// the next one.
	var groups [][]SearchPair
	var cur, waiting []SearchPair
	seen := map[string]bool{}
	closeGroup := func() {
		if len(cur) > 0 {
			groups = append(groups, cur)
		}
		cur, seen = nil, map[string]bool{}
	}
	for k := 0; k < len(order) || len(waiting) > 0; {
		var p SearchPair
		switch {
		case len(waiting) > 0 && !seen[qid(waiting[0])]:
			p, waiting = waiting[0], waiting[1:]
		case k < len(order):
			p = pairs[order[k]]
			k++
			if seen[qid(p)] {
				waiting = append(waiting, p)
				continue
			}
		default:
			closeGroup() // only pairs of queries already in the group remain
			continue
		}
		seen[qid(p)] = true
		cur = append(cur, p)
		if len(cur) == size {
			closeGroup()
		}
	}
	closeGroup()
	// A group of one pair cannot compare its query with another: its pair
	// joins an earlier group that does not hold its query.
	var kept [][]SearchPair
	for _, g := range groups {
		if len(g) >= 2 {
			kept = append(kept, g)
		}
	}
	// Single pairs of different queries first make groups together.
	var singles, alone []SearchPair
	for _, g := range groups {
		if len(g) == 1 {
			singles = append(singles, g[0])
		}
	}
	for len(singles) > 0 {
		g, used := []SearchPair{singles[0]}, map[string]bool{qid(singles[0]): true}
		var rest []SearchPair
		for _, p := range singles[1:] {
			if len(g) < size && !used[qid(p)] {
				g, used[qid(p)] = append(g, p), true
			} else {
				rest = append(rest, p)
			}
		}
		if len(g) >= 2 {
			kept = append(kept, g)
		} else {
			alone = append(alone, g[0])
		}
		singles = rest
	}
	// A pair still alone joins a group without its query, or else takes
	// one pair from a group of more than two to form a new one.
	for _, p := range alone {
		placed := false
		for i := 0; i < len(kept) && !placed; i++ {
			has := false
			for _, q := range kept[i] {
				has = has || qid(q) == qid(p)
			}
			if !has && len(kept[i]) < size {
				kept[i], placed = append(kept[i], p), true
			}
		}
		for i := 0; i < len(kept) && !placed; i++ {
			for j, q := range kept[i] {
				if len(kept[i]) > 2 && qid(q) != qid(p) {
					kept[i] = append(kept[i][:j:j], kept[i][j+1:]...)
					kept, placed = append(kept, []SearchPair{q, p}), true
					break
				}
			}
		}
		// Else it forms a group with a pair of another query, used twice: a
		// pair seen twice is harmless, a pair dropped is a lost example. Only
		// a set where every pair has the same query cannot place it.
		for i := 0; i < len(pairs) && !placed; i++ {
			if q := pairs[i]; qid(q) != qid(p) {
				kept, placed = append(kept, []SearchPair{q, p}), true
			}
		}
	}

	out := make([]ChoiceBatch, 0, len(kept))
	for _, cur := range kept {
		var b ChoiceBatch
		var ids []string
		for _, p := range cur {
			b.Candidates = append(b.Candidates, Candidate{Name: m.info.Prompts["document"] + p.Document})
			ids = append(ids, did(p))
		}
		for _, p := range cur {
			for _, n := range p.Negatives {
				b.Candidates = append(b.Candidates, Candidate{Name: m.info.Prompts["document"] + n})
				ids = append(ids, cmp.Or(byText[n], n))
			}
		}
		for i, p := range cur {
			b.Texts = append(b.Texts, m.info.Prompts["query"]+p.Query)
			c := []int{i}
			for j, id := range ids {
				if j != i && relevant[qid(p)][id] {
					c = append(c, j)
				}
			}
			b.Correct = append(b.Correct, c)
		}
		out = append(out, b)
	}
	return out
}
