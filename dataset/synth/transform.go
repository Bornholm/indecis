package synth

import (
	"encoding/base64"
	"encoding/hex"
	"math/rand"
	"strings"
	"unicode"
)

// TransformFunc rewrites a rendered text. The generator provides its
// rng: random transformations stay reproducible.
type TransformFunc func(s string, rng *rand.Rand) string

// Transforms lists the transformations for {{x:name}}...{{/x}}. They
// reproduce the common disguises of a text meant to slip past filters;
// a caller can add more before parsing its templates.
var Transforms = map[string]TransformFunc{
	"upper":  func(s string, _ *rand.Rand) string { return strings.ToUpper(s) },
	"lower":  func(s string, _ *rand.Rand) string { return strings.ToLower(s) },
	"base64": func(s string, _ *rand.Rand) string { return base64.StdEncoding.EncodeToString([]byte(s)) },
	"hex":    func(s string, _ *rand.Rand) string { return hex.EncodeToString([]byte(s)) },
	"rot13":  func(s string, _ *rand.Rand) string { return strings.Map(rot13, s) },
	// spaced separates each letter with a space: "i g n o r e".
	"spaced": func(s string, _ *rand.Rand) string {
		var b strings.Builder
		for i, r := range s {
			if i > 0 && r != ' ' {
				b.WriteByte(' ')
			}
			b.WriteRune(r)
		}
		return b.String()
	},
	// leet replaces some letters with digits.
	"leet": func(s string, rng *rand.Rand) string {
		table := map[rune]rune{'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'A': '4', 'E': '3', 'I': '1', 'O': '0', 'S': '5', 'T': '7'}
		return strings.Map(func(r rune) rune {
			if l, ok := table[r]; ok && rng.Float64() < 0.7 {
				return l
			}
			return r
		}, s)
	},
	// homoglyph replaces Latin letters with their Cyrillic lookalikes.
	"homoglyph": func(s string, rng *rand.Rand) string {
		table := map[rune]rune{'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у', 'A': 'А', 'C': 'С', 'E': 'Е', 'O': 'О', 'P': 'Р'}
		return strings.Map(func(r rune) rune {
			if l, ok := table[r]; ok && rng.Float64() < 0.5 {
				return l
			}
			return r
		}, s)
	},
	// zwsp slips zero-width spaces inside words.
	"zwsp": func(s string, rng *rand.Rand) string {
		var b strings.Builder
		for _, r := range s {
			b.WriteRune(r)
			if unicode.IsLetter(r) && rng.Float64() < 0.3 {
				b.WriteRune('​')
			}
		}
		return b.String()
	},
	// noise reproduces the intra-word spacing of text extracted from
	// PDFs (taken from go-anon): "s ite v is it".
	"noise": func(s string, rng *rand.Rand) string {
		var b strings.Builder
		for _, r := range s {
			b.WriteRune(r)
			if r != ' ' && r != '\n' && rng.Float64() < 0.35 {
				b.WriteByte(' ')
			}
		}
		return b.String()
	},
}

func rot13(r rune) rune {
	switch {
	case r >= 'a' && r <= 'z':
		return 'a' + (r-'a'+13)%26
	case r >= 'A' && r <= 'Z':
		return 'A' + (r-'A'+13)%26
	}
	return r
}
