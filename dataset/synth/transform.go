package synth

import (
	"encoding/base64"
	"encoding/hex"
	"math/rand"
	"strings"
	"unicode"
)

// TransformFunc réécrit un texte rendu. Le générateur fournit son rng : les
// transformations aléatoires restent reproductibles.
type TransformFunc func(s string, rng *rand.Rand) string

// Transforms recense les transformations de {{x:nom}}…{{/x}}. Elles
// reproduisent les déguisements courants d'un texte qu'on veut faire passer
// sous les filtres ; un appelant peut en ajouter avant de parser ses gabarits.
var Transforms = map[string]TransformFunc{
	"upper":  func(s string, _ *rand.Rand) string { return strings.ToUpper(s) },
	"lower":  func(s string, _ *rand.Rand) string { return strings.ToLower(s) },
	"base64": func(s string, _ *rand.Rand) string { return base64.StdEncoding.EncodeToString([]byte(s)) },
	"hex":    func(s string, _ *rand.Rand) string { return hex.EncodeToString([]byte(s)) },
	"rot13":  func(s string, _ *rand.Rand) string { return strings.Map(rot13, s) },
	// spaced sépare chaque lettre d'une espace : « i g n o r e ».
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
	// leet remplace une partie des lettres par des chiffres.
	"leet": func(s string, rng *rand.Rand) string {
		table := map[rune]rune{'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'A': '4', 'E': '3', 'I': '1', 'O': '0', 'S': '5', 'T': '7'}
		return strings.Map(func(r rune) rune {
			if l, ok := table[r]; ok && rng.Float64() < 0.7 {
				return l
			}
			return r
		}, s)
	},
	// homoglyph remplace des lettres latines par leurs sosies cyrilliques.
	"homoglyph": func(s string, rng *rand.Rand) string {
		table := map[rune]rune{'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у', 'A': 'А', 'C': 'С', 'E': 'Е', 'O': 'О', 'P': 'Р'}
		return strings.Map(func(r rune) rune {
			if l, ok := table[r]; ok && rng.Float64() < 0.5 {
				return l
			}
			return r
		}, s)
	},
	// zwsp glisse des espaces de largeur nulle à l'intérieur des mots.
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
	// noise reproduit l'espacement intra-mot du texte extrait des PDF
	// (repris de go-anon) : « V is ite te c hniq ue ».
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
