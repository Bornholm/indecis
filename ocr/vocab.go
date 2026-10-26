package ocr

import (
	"fmt"
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
	"«»“”‘’„–—…•°€£¥§©®™×÷±²³¹½¼¾µ‰†·¿¡№"

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
// spaces of all widths become one space, ligatures their letters, the ends
// trimmed. CTC cannot read two spaces in a row anyway.
func NormalizeText(s string) string {
	s = textReplacer.Replace(s)
	return strings.Join(strings.Fields(s), " ")
}

var textReplacer = strings.NewReplacer(
	" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "\t", " ",
	"ﬁ", "fi", "ﬂ", "fl", "ﬀ", "ff", "ﬃ", "ffi", "ﬄ", "ffl", "­", "",
)
