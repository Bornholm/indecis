package indecis

import (
	"fmt"

	"github.com/bornholm/indecis/tokenizer"
)

// PruneStats describes a vocabulary pruning.
type PruneStats struct {
	Before, After int // tokens before and after
}

// PruneVocabulary reduces the vocabulary to the tokens that the corpus
// tokenization uses, including BPE intermediates, seen at least minCount
// times (1: all that appear). The embedding table, 93% of a small model's
// weights, shrinks accordingly: for use in a few languages, a multilingual
// vocabulary of 256,000 tokens reduces to a few tens of thousands.
//
// A corpus text then tokenizes exactly as before; a text that would have
// needed a removed token tokenizes into smaller pieces, at worst into bytes.
// The corpus must therefore represent the texts the model will judge
// (languages, domains), and the effect is measured on texts outside the
// corpus (Evaluate before and after). The model is modified in place; Save
// writes the reduced tokenizer.
func (m *Model) PruneVocabulary(corpus []Input, minCount int) (PruneStats, error) {
	src, err := m.tokenizerSource()
	if err != nil {
		return PruneStats{}, err
	}
	count := make([]int, m.tok.VocabSize())
	visit := func(id int32) { count[id]++ }
	for _, in := range corpus {
		m.tok.Trace(in.Text, visit)
		if in.Context != "" {
			m.tok.Trace(in.Context, visit)
		}
	}
	out, newID, err := tokenizer.Prune(src, func(id int32) bool { return count[id] >= max(1, minCount) })
	if err != nil {
		return PruneStats{}, err
	}
	tok, err := tokenizer.Parse(out)
	if err != nil {
		return PruneStats{}, err
	}
	if len(newID) != m.enc.Cfg.Vocab {
		return PruneStats{}, fmt.Errorf("indecis: tokenizer (%d ids) does not match model (%d)", len(newID), m.enc.Cfg.Vocab)
	}
	st := PruneStats{Before: m.enc.Cfg.Vocab, After: tok.VocabSize()}
	if err := m.enc.RemapVocabulary(newID, tok.VocabSize()); err != nil {
		return PruneStats{}, err
	}
	m.tok, m.tokenizerJSON = tok, out
	m.embedCache.clear()
	return st, nil
}
