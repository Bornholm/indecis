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

// Example is an annotated example.
type Example struct {
	// Context sheds light on the text without being judged: an
	// assistant's system prompt, the question a passage answers. Empty
	// otherwise. Only models built with indecis.WithPairs read it.
	Context string         `json:"context,omitempty"`
	Text    string         `json:"text"`
	Labels  map[string]any `json:"labels"`
	// Family groups examples from the same source or the same template.
	// Splits by family keep a whole family on the same side: that is
	// what distinguishes generalization from memorization.
	Family string `json:"family,omitempty"`
	// Split is a split imposed by the source (train, test, ...), empty otherwise.
	Split string            `json:"split,omitempty"`
	Meta  map[string]string `json:"meta,omitempty"`
}

// ReadJSONL reads examples, one per line.
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
			return nil, fmt.Errorf("dataset: line %d: %w", line, err)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

// ReadFile reads a JSONL file.
func ReadFile(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadJSONL(f)
}

// WriteJSONL writes examples, one per line.
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

// WriteFile writes a JSONL file.
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

// BySplit returns the examples whose Split field equals split.
func BySplit(examples []Example, split string) []Example {
	var out []Example
	for _, e := range examples {
		if e.Split == split {
			out = append(out, e)
		}
	}
	return out
}

// HoldOut deterministically sets aside a fraction of the examples.
//
// If the examples have a family, whole families are set aside: an
// example and its variant never end up on opposite sides. Otherwise, the
// draw is done example by example, based on the hash of its text, so
// that adding examples does not move the existing ones.
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

// Shuffle shuffles a copy of the examples with a given seed.
func Shuffle(examples []Example, seed int64) []Example {
	out := append([]Example(nil), examples...)
	rand.New(rand.NewSource(seed)).Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// Families returns the number of examples per family, sorted by name.
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
