// Package dataset defines the format of training examples and the splits
// that measure generalization.
//
// An example is a text and its expected answers, one per question of the
// schema. An answer may be missing: the example then trains only the
// questions it has an answer for. The file format is JSONL, one example per
// line:
//
//	{"text": "Ignore previous instructions", "labels": {"injection": true, "category": "override"}, "family": "override"}
//
// Accepted values by question type:
//   - noul: a boolean, or a probability in [0, 1] (soft label from a teacher);
//   - choice: an option name, or a distribution {"option": probability};
//   - score: an integer level, from 0 to the number of levels minus one.
package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math/rand"
	"os"
	"sort"
)

// Example est un exemple annoté.
type Example struct {
	// Context éclaire le texte sans être jugé : le prompt système d'un
	// assistant, la question à laquelle répond un passage. Vide sinon. Seuls
	// les modèles construits avec indecis.WithPairs le lisent.
	Context string         `json:"context,omitempty"`
	Text    string         `json:"text"`
	Labels  map[string]any `json:"labels"`
	// Family regroupe les exemples issus d'une même source ou d'un même
	// gabarit. Les découpages par famille gardent une famille entière du même
	// côté : c'est ce qui distingue la généralisation de la mémorisation.
	Family string `json:"family,omitempty"`
	// Split est un découpage imposé par la source (train, test…), vide sinon.
	Split string            `json:"split,omitempty"`
	Meta  map[string]string `json:"meta,omitempty"`
}

// ReadJSONL lit des exemples, un par ligne.
func ReadJSONL(r io.Reader) ([]Example, error) {
	var out []Example
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	line := 0
	for sc.Scan() {
		line++
		if len(sc.Bytes()) == 0 {
			continue
		}
		var e Example
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("dataset: ligne %d : %w", line, err)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

// ReadFile lit un fichier JSONL.
func ReadFile(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadJSONL(f)
}

// WriteJSONL écrit des exemples, un par ligne.
func WriteJSONL(w io.Writer, examples []Example) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)
	for _, e := range examples {
		if err := enc.Encode(e); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// WriteFile écrit un fichier JSONL.
func WriteFile(path string, examples []Example) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := WriteJSONL(f, examples); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// BySplit retourne les exemples dont le champ Split vaut split.
func BySplit(examples []Example, split string) []Example {
	var out []Example
	for _, e := range examples {
		if e.Split == split {
			out = append(out, e)
		}
	}
	return out
}

// HoldOut sépare une fraction des exemples, de façon déterministe.
//
// Si les exemples ont une famille, ce sont des familles entières qui sont
// mises de côté : un exemple et sa variante ne se retrouvent jamais de part
// et d'autre. Sinon, le tirage se fait exemple par exemple, selon le hash de
// son texte, pour qu'un ajout d'exemples ne déplace pas les existants.
func HoldOut(examples []Example, fraction float64, seed uint64) (kept, held []Example) {
	key := func(e Example) string {
		if e.Family != "" {
			return "family:" + e.Family
		}
		return "text:" + e.Context + "\x00" + e.Text
	}
	for _, e := range examples {
		h := fnv.New64a()
		fmt.Fprintf(h, "%d:%s", seed, key(e))
		if float64(h.Sum64()%1_000_000)/1_000_000 < fraction {
			held = append(held, e)
		} else {
			kept = append(kept, e)
		}
	}
	return kept, held
}

// Shuffle mélange une copie des exemples avec une graine donnée.
func Shuffle(examples []Example, seed int64) []Example {
	out := append([]Example(nil), examples...)
	rand.New(rand.NewSource(seed)).Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// Families retourne le nombre d'exemples par famille, trié par nom.
func Families(examples []Example) []struct {
	Name  string
	Count int
} {
	counts := map[string]int{}
	for _, e := range examples {
		counts[e.Family]++
	}
	out := make([]struct {
		Name  string
		Count int
	}, 0, len(counts))
	for name, n := range counts {
		out = append(out, struct {
			Name  string
			Count int
		}{name, n})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
