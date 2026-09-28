// Gazetteers: weighted lists of values, taken from go-anon.
//
// The format is a TSV "value <TAB> weight <TAB> metadata(JSON)", chosen to
// stay readable and diffable. Raw weights from statistical sources are
// very spiky: a handful of values overwhelm the long tail. The
// flattening exponent makes this trade-off tunable.
package synth

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// Entry is a gazetteer value with its weight and metadata.
type Entry struct {
	Value    string
	Weight   float64
	Metadata map[string]string
}

// Gazetteer is a weighted list ready for sampling.
type Gazetteer struct {
	entries []Entry
	cum     []float64 // cumulative weights, for binary search
	total   float64
}

// GazetteerOptions drives the shaping of the distribution at load time.
type GazetteerOptions struct {
	// Alpha flattens the weights: w' = w^Alpha. 1 keeps the real
	// distribution, 0 makes it uniform. Default 0.6.
	Alpha float64
	// MinWeight discards values below this raw weight, to cut the
	// spelling noise of source files.
	MinWeight float64
}

// DefaultGazetteerOptions returns the recommended settings.
func DefaultGazetteerOptions() GazetteerOptions { return GazetteerOptions{Alpha: 0.6} }

// NewGazetteer builds a Gazetteer from already-loaded entries.
func NewGazetteer(entries []Entry, opts GazetteerOptions) (*Gazetteer, error) {
	if opts.Alpha <= 0 {
		opts.Alpha = 1
	}
	s := &Gazetteer{}
	for _, e := range entries {
		if e.Weight < opts.MinWeight || e.Weight <= 0 {
			continue
		}
		w := math.Pow(e.Weight, opts.Alpha)
		s.total += w
		s.entries = append(s.entries, e)
		s.cum = append(s.cum, s.total)
	}
	if len(s.entries) == 0 {
		return nil, fmt.Errorf("gazetteer empty after filtering (MinWeight=%v)", opts.MinWeight)
	}
	return s, nil
}

// LoadGazetteer reads a gazetteer in TSV format. Empty lines and those
// starting with "#" are ignored. A missing weight defaults to 1.
func LoadGazetteer(r io.Reader, opts GazetteerOptions) (*Gazetteer, error) {
	var entries []Entry
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimRight(sc.Text(), "\r")
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		cols := strings.Split(raw, "\t")
		e := Entry{Value: strings.TrimSpace(cols[0]), Weight: 1}
		if e.Value == "" {
			continue
		}
		if len(cols) > 1 && strings.TrimSpace(cols[1]) != "" {
			w, err := strconv.ParseFloat(strings.TrimSpace(cols[1]), 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid weight %q: %w", line, cols[1], err)
			}
			e.Weight = w
		}
		if len(cols) > 2 && strings.TrimSpace(cols[2]) != "" {
			if err := json.Unmarshal([]byte(cols[2]), &e.Metadata); err != nil {
				return nil, fmt.Errorf("line %d: invalid metadata: %w", line, err)
			}
		}
		entries = append(entries, e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return NewGazetteer(entries, opts)
}

// Subset builds a Gazetteer restricted to the entries satisfying pred,
// keeping their already-flattened weights.
//
// Used to constrain semantic coherence: we do not compose a "Public
// Works Laboratory". Returns nil if no entry passes, leaving it to the
// caller to fall back to the full Gazetteer.
func (s *Gazetteer) Subset(pred func(Entry) bool) *Gazetteer {
	out := &Gazetteer{}
	for i, e := range s.entries {
		if !pred(e) {
			continue
		}
		w := s.cum[i]
		if i > 0 {
			w -= s.cum[i-1]
		}
		out.total += w
		out.entries = append(out.entries, e)
		out.cum = append(out.cum, out.total)
	}
	if len(out.entries) == 0 {
		return nil
	}
	return out
}

// Pick draws an entry according to the weighted distribution.
func (s *Gazetteer) Pick(rng *rand.Rand) Entry {
	target := rng.Float64() * s.total
	i := sort.SearchFloat64s(s.cum, target)
	if i >= len(s.entries) {
		i = len(s.entries) - 1
	}
	return s.entries[i]
}

// PickValue draws an entry and returns only its value.
func (s *Gazetteer) PickValue(rng *rand.Rand) string { return s.Pick(rng).Value }

// Len returns the number of retained entries.
func (s *Gazetteer) Len() int { return len(s.entries) }
