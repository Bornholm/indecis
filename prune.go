package indecis

import (
	"fmt"

	"github.com/bornholm/indecis/tokenizer"
)

// PruneStats décrit un élagage du vocabulaire.
type PruneStats struct {
	Before, After int // tokens avant et après
}

// PruneVocabulary réduit le vocabulaire aux tokens que le découpage de
// corpus utilise, intermédiaires du BPE compris, vus au moins minCount fois
// (1 : tous ceux qui apparaissent). La table d'embeddings, 93 % des poids
// des petits modèles, rétrécit d'autant : pour un usage en quelques
// langues, un vocabulaire multilingue de 256 000 tokens se réduit à
// quelques dizaines de milliers.
//
// Un texte du corpus se découpe ensuite exactement comme avant ; un texte
// qui aurait eu besoin d'un token retiré se découpe en morceaux plus
// petits, au pire en octets. Le corpus doit donc représenter les textes
// que le modèle jugera (langues, domaines), et l'effet se mesure sur des
// textes hors corpus (Evaluate avant et après). Le modèle est modifié en
// place ; Save écrit le tokenizer réduit.
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
		return PruneStats{}, fmt.Errorf("indecis: le tokenizer (%d ids) ne correspond pas au modèle (%d)", len(newID), m.enc.Cfg.Vocab)
	}
	st := PruneStats{Before: m.enc.Cfg.Vocab, After: tok.VocabSize()}
	if err := m.enc.RemapVocabulary(newID, tok.VocabSize()); err != nil {
		return PruneStats{}, err
	}
	m.tok, m.tokenizerJSON = tok, out
	m.embedCache.clear()
	return st, nil
}
