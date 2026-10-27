package ocr

import (
	"fmt"
	"math/rand"
	"strings"
	"unicode/utf8"
)

// DefaultCharset is the set of characters of the French and English
// printed recognizer: printable ASCII, the letters of French and of the
// neighbouring Latin-script languages, and the typography of French
// documents.
const DefaultCharset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~" +
	"àâäæçéèêëîïôöœùûüÿÀÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ" +
	"áíóúñãõåøßìòÁÍÓÚÑÃÕÅØÌÒ" +
	"«»“”‘’„–—…•°€£¥§©®™×÷±²³¹½¼¾µ‰†·¿¡№" +
	// Extended (models from v5 on): superscripts and subscripts, bullets
	// and boxes, arrows, Greek and math signs, letters of Central Europe
	// and Turkish.
	"⁰⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉■□▪●○◆◇►▶✓✔✗☐☑←→↑↓⇒≤≥≠≈∞√∑∆∂ΩαβγδεθλπσφωΔΣ′″⅓⅔℃¢" +
	"čšžřěůďťňĺľąęłńśźżćőűČŠŽŘĚŮĎŤŇĄĘŁŃŚŹŻĆŐŰğşıİĞŞ"

// NewVocab builds a vocabulary from a charset, one symbol per character,
// in the charset's order. Duplicates are an error.
func NewVocab(charset string) (Vocab, error) {
	v := Vocab{Blank: 0}
	seen := map[rune]bool{}
	for _, r := range charset {
		if seen[r] {
			return v, fmt.Errorf("ocr: %q appears twice in the charset", r)
		}
		seen[r] = true
		v.Chars = append(v.Chars, string(r))
	}
	return v, nil
}

// Encoder maps texts to symbol ids.
type Encoder struct {
	ids    map[string]int
	maxLen int // longest symbol, in runes
}

// Encoder returns an encoder for v. Symbols of several characters (some
// vocabularies have them) are matched longest first.
func (v Vocab) Encoder() *Encoder {
	e := &Encoder{ids: make(map[string]int, len(v.Chars)), maxLen: 1}
	for i, c := range v.Chars {
		e.ids[c] = i + 1
		e.maxLen = max(e.maxLen, utf8.RuneCountInString(c))
	}
	return e
}

// Encode returns the symbol ids of text, or false and the first character
// outside the vocabulary.
func (e *Encoder) Encode(text string) ([]int, rune, bool) {
	var ids []int
	runes := []rune(text)
	for i := 0; i < len(runes); {
		found := false
		for n := min(e.maxLen, len(runes)-i); n >= 1; n-- {
			if id, ok := e.ids[string(runes[i:i+n])]; ok {
				ids = append(ids, id)
				i += n
				found = true
				break
			}
		}
		if !found {
			return nil, runes[i], false
		}
	}
	return ids, 0, true
}

// NormalizeText puts a reference text in the form the recognizer reads:
// letters precomposed (a base letter followed by a combining accent, as
// some transcriptions write them, becomes one letter), spaces of all widths
// one space, ligatures their letters, the "¬" some transcriptions write
// for a printed end-of-line hyphen a hyphen, the ends trimmed. CTC cannot
// read two spaces in a row anyway.
func NormalizeText(s string) string {
	s = textReplacer.Replace(Compose(s))
	return strings.Join(strings.Fields(s), " ")
}

// Compose precomposes Latin letters followed by combining marks (the
// canonical compositions of composeTable, applied repeatedly: "e" + U+0302
// + U+0301 gives "ế").
func Compose(s string) string {
	if !hasCombining(s) {
		return s
	}
	out := make([]rune, 0, len(s))
	for _, r := range s {
		if n := len(out); n > 0 {
			if c, ok := composeTable[[2]rune{out[n-1], r}]; ok {
				out[n-1] = c
				continue
			}
		}
		out = append(out, r)
	}
	return string(out)
}

func hasCombining(s string) bool {
	for _, r := range s {
		if r >= 0x300 && r <= 0x36F {
			return true
		}
	}
	return false
}

var textReplacer = strings.NewReplacer(
	" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "\t", " ",
	"ﬁ", "fi", "ﬂ", "fl", "ﬀ", "ff", "ﬃ", "ffi", "ﬄ", "ffl", "­", "", "¬", "-",
)

// ExtendHead maps a recognizer's output head onto a new vocabulary: the
// rows of the symbols both vocabularies hold are kept, the others drawn
// small (uniform in ±0.01), so that training goes on from the old model
// with new symbols to learn. head is [old symbols, dim], bias [old].
func ExtendHead(head, bias []float32, dim int, old, new Vocab, seed int64) ([]float32, []float32) {
	r := rand.New(rand.NewSource(seed))
	idx := map[string]int{}
	for i, c := range old.Chars {
		idx[c] = i + 1
	}
	V := len(new.Chars) + 1
	h, b := make([]float32, V*dim), make([]float32, V)
	copy(h[:dim], head[:dim]) // blank
	b[0] = bias[0]
	for i, c := range new.Chars {
		row := h[(i+1)*dim : (i+2)*dim]
		if j, ok := idx[c]; ok {
			copy(row, head[j*dim:(j+1)*dim])
			b[i+1] = bias[j]
			continue
		}
		for k := range row {
			row[k] = (2*r.Float32() - 1) * 0.01
		}
		b[i+1] = bias[0] - 4 // new symbols start unlikely
	}
	return h, b
}
