package tokenizer

import (
	"encoding/json"
	"encoding/json/jsontext"
	"fmt"
	"io"
	"strings"
)

// decode lit un tokenizer.json en flux. Le vocabulaire (256 000 entrées) et
// les fusions sont lus entrée par entrée : ni le fichier (34 Mo) ni une map
// du vocabulaire ne passent en mémoire. Le reste du fichier, petit, est
// décodé normalement dans f.
func decode(r io.Reader) (f fileJSON, byID []string, pairs [][2]string, err error) {
	d := jsontext.NewDecoder(r, jsontext.AllowDuplicateNames(true))
	if err = expect(d, '{'); err != nil {
		return
	}
	for d.PeekKind() != '}' {
		var key string
		if key, err = readString(d); err != nil {
			return
		}
		switch key {
		case "model":
			if byID, pairs, err = decodeModel(d, &f); err != nil {
				return
			}
		case "added_tokens", "normalizer", "pre_tokenizer", "post_processor":
			var v jsontext.Value
			if v, err = d.ReadValue(); err != nil {
				return
			}
			if err = json.Unmarshal(v, fieldOf(&f, key)); err != nil {
				return
			}
		default:
			if err = d.SkipValue(); err != nil {
				return
			}
		}
	}
	err = expect(d, '}')
	return
}

func fieldOf(f *fileJSON, key string) any {
	switch key {
	case "added_tokens":
		return &f.AddedTokens
	case "normalizer":
		return &f.Normalizer
	case "pre_tokenizer":
		return &f.PreTokenizer
	}
	return &f.PostProcessor
}

func decodeModel(d *jsontext.Decoder, f *fileJSON) (byID []string, pairs [][2]string, err error) {
	if err = expect(d, '{'); err != nil {
		return
	}
	rest := map[string]jsontext.Value{}
	for d.PeekKind() != '}' {
		var key string
		if key, err = readString(d); err != nil {
			return
		}
		switch key {
		case "vocab":
			if byID, err = decodeVocab(d); err != nil {
				return
			}
		case "merges":
			if pairs, err = decodeMerges(d); err != nil {
				return
			}
		default:
			v, e := d.ReadValue()
			if e != nil {
				return nil, nil, e
			}
			rest[key] = v.Clone()
		}
	}
	if err = expect(d, '}'); err != nil {
		return
	}
	b, err := json.Marshal(rest)
	if err != nil {
		return
	}
	err = json.Unmarshal(b, &f.Model)
	return
}

// decodeVocab lit {"chaîne": id, …} dans byID[id] = chaîne.
func decodeVocab(d *jsontext.Decoder) ([]string, error) {
	if err := expect(d, '{'); err != nil {
		return nil, err
	}
	var byID []string
	for d.PeekKind() != '}' {
		s, err := readString(d)
		if err != nil {
			return nil, err
		}
		tok, err := d.ReadToken()
		if err != nil {
			return nil, err
		}
		if tok.Kind() != '0' {
			return nil, fmt.Errorf("tokenizer: vocab : id attendu pour %q", s)
		}
		id, err := tok.Int()
		if err != nil || id < 0 || id >= 1<<24 {
			return nil, fmt.Errorf("tokenizer: vocab : id %d invalide", id)
		}
		for int(id) >= len(byID) {
			byID = append(byID, "")
		}
		if byID[id] != "" {
			return nil, fmt.Errorf("tokenizer: id %d attribué deux fois", id)
		}
		if s == "" {
			return nil, fmt.Errorf("tokenizer: vocab : chaîne vide")
		}
		byID[id] = s
	}
	return byID, expect(d, '}')
}

// decodeMerges accepte les deux formats : ["a","b"] et "a b".
func decodeMerges(d *jsontext.Decoder) ([][2]string, error) {
	if err := expect(d, '['); err != nil {
		return nil, err
	}
	var pairs [][2]string
	for d.PeekKind() != ']' {
		switch d.PeekKind() {
		case '"':
			s, err := readString(d)
			if err != nil {
				return nil, err
			}
			a, b, ok := strings.Cut(s, " ")
			if !ok {
				return nil, fmt.Errorf("tokenizer: merge invalide %q", s)
			}
			pairs = append(pairs, [2]string{a, b})
		case '[':
			d.ReadToken()
			a, err := readString(d)
			if err != nil {
				return nil, err
			}
			b, err := readString(d)
			if err != nil {
				return nil, err
			}
			if err := expect(d, ']'); err != nil {
				return nil, err
			}
			pairs = append(pairs, [2]string{a, b})
		default:
			return nil, fmt.Errorf("tokenizer: merges : élément inattendu")
		}
	}
	return pairs, expect(d, ']')
}

func readString(d *jsontext.Decoder) (string, error) {
	tok, err := d.ReadToken()
	if err != nil {
		return "", fmt.Errorf("tokenizer: %w", err)
	}
	if tok.Kind() != '"' {
		return "", fmt.Errorf("tokenizer: chaîne attendue, %v lu", tok.Kind())
	}
	return tok.String(), nil
}

func expect(d *jsontext.Decoder, k jsontext.Kind) error {
	tok, err := d.ReadToken()
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	if tok.Kind() != k {
		return fmt.Errorf("tokenizer: %v attendu, %v lu", k, tok.Kind())
	}
	return nil
}
