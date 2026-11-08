package modernbert

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bornholm/indecis/tokenizer"
)

// granite-embedding-311m-multilingual-r2: 22 layers of 768, CLS pooling,
// Gemma 3 vocabulary. Fixtures from tools/oracle/forward_fixtures.py
// --very-long 1500 (the last case is past 4,000 tokens).
func loadGranite(t testing.TB) (*Model, *tokenizer.Tokenizer) {
	t.Helper()
	dir := os.Getenv("INDECIS_GRANITE_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/granite-embedding-311m-multilingual-r2")
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("granite model missing (%s): set INDECIS_GRANITE_DIR", dir)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	return m, tok
}

// cosDist is 1 - cos(a, b).
func cosDist(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return 1 - ab/math.Sqrt(aa*bb)
}

// relDiff is max|a-b| relative to max|b|: the states of a deep encoder
// reach a few tens, an absolute tolerance would not mean the same thing.
func relDiff(a, b []float32) float64 {
	var m float64
	for _, v := range b {
		m = max(m, math.Abs(float64(v)))
	}
	return maxAbsDiff(a, b) / m
}

func TestGraniteParity(t *testing.T) {
	m, tok := loadGranite(t)
	if !m.Cfg.CLS() {
		t.Fatalf("pooling %q, expected cls (1_Pooling)", m.Cfg.Pooling)
	}
	b, err := os.ReadFile("../../testdata/granite311m/forward_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx forwardFixtures
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	H := m.Cfg.Hidden
	var seqs [][]int32
	for i, c := range fx.Cases {
		name := fmt.Sprintf("%d:%.30q", i, c.Text)
		ids := tok.Encode(c.Text)
		if !slices.Equal(ids, c.IDs) {
			t.Fatalf("%s: tokens %v, expected %v", name, ids, c.IDs)
		}
		seqs = append(seqs, ids)
		got, err := m.Encode(NewBatch([][]int32{ids}, m.Cfg.PadID))
		if err != nil {
			t.Fatal(err)
		}
		rel, cd := relDiff(got, c.Pooled), cosDist(got, c.Pooled)
		t.Logf("%s: %d tokens, Encode rel|Δ| = %.2g, 1-cos = %.2g", name, len(ids), rel, cd)
		if rel > 1e-4 || cd > 1e-7 {
			t.Errorf("%s: Encode rel|Δ| = %g, 1-cos = %g", name, rel, cd)
		}
		if len(ids) > 512 {
			continue // Forward keeps every activation: short cases only
		}
		bt := NewBatch([][]int32{ids}, m.Cfg.PadID)
		s, err := m.Forward(bt)
		if err != nil {
			t.Fatal(err)
		}
		if rel := relDiff(m.Pool(s.Hidden, bt), c.Pooled); rel > 1e-4 {
			t.Errorf("%s: Forward rel|Δ| = %g", name, rel)
		}
		for _, h := range fx.Hidden {
			if h.Case == i {
				want := readF32(t, fmt.Sprintf("../../testdata/granite311m/forward_hidden_%d.f32", i))
				if rel := relDiff(s.Hidden, want); rel > 5e-4 {
					t.Errorf("%s: hidden rel|Δ| = %g over %d tokens", name, rel, h.Rows)
				}
			}
		}
	}
	// Batched: CLS takes each sequence's own first token.
	got, err := m.Encode(NewBatch(seqs, m.Cfg.PadID))
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range fx.Cases {
		if rel := relDiff(got[i*H:(i+1)*H], c.Pooled); rel > 1e-4 {
			t.Errorf("batched case %d: rel|Δ| = %g", i, rel)
		}
	}
}
