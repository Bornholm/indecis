package linerec

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// indraDir holds Indra OCR's recognizer converted by
// tools/oracle/ocr_indra_check.py. Its weights are not distributable: the
// tests that need them are skipped without them.
func indraDir(t testing.TB) string {
	dir := os.Getenv("INDECIS_INDRA_REC_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/indra-rec")
	}
	if _, err := os.Stat(filepath.Join(dir, "check.json")); err != nil {
		t.Skip("Indra recognizer missing (tools/oracle/ocr_indra_check.py)")
	}
	return dir
}

type indraCheck struct {
	Texts  []string `json:"texts"`
	Widths []int    `json:"widths"`
}

// indraLines returns the reference lines, each [Rows, w].
func indraLines(t testing.TB, dir string, rows int) (indraCheck, [][]float32) {
	b, err := os.ReadFile(filepath.Join(dir, "check.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c indraCheck
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	x := readF32(t, filepath.Join(dir, "check_x.f32"))
	var lines [][]float32
	off := 0
	for _, w := range c.Widths {
		wp := (w + 7) / 8 * 8
		ink := make([]float32, rows*w)
		for y := range rows {
			copy(ink[y*w:(y+1)*w], x[off+y*wp:])
		}
		lines = append(lines, ink)
		off += rows * wp
	}
	return c, lines
}

func TestIndraMatchesPyTorch(t *testing.T) {
	dir := indraDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		t.Fatal(err)
	}
	var voc struct {
		Chars []string `json:"chars"`
	}
	if err := json.Unmarshal(b, &voc); err != nil {
		t.Fatal(err)
	}
	c, lines := indraLines(t, dir, m.Cfg.Rows())
	want := readF32(t, filepath.Join(dir, "check_logits.f32"))
	V := m.Cfg.Vocab
	var s State
	off := 0
	for i, w := range c.Widths {
		m.Forward(&s, lines[i], w)
		n := s.L * V
		d := maxDiff(s.Logits[:n], want[off:off+n])
		if d > 1e-3 {
			t.Errorf("line %d: max |Δ logits| = %.2e", i, d)
		}
		off += n
		syms, conf := Greedy(s.Logits, s.L, V)
		var sb strings.Builder
		for _, sy := range syms {
			sb.WriteString(voc.Chars[sy.ID-1])
		}
		t.Logf("%q (confidence %.3f, max |Δ logits| %.1e)", sb.String(), conf, d)
	}
}

// BenchmarkIndraForward reads the reference lines (rendered text), and
// random ink at fixed widths.
func BenchmarkIndraForward(b *testing.B) {
	dir := indraDir(b)
	m, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	if os.Getenv("INDECIS_INT8") == "1" {
		m.PrepareInt8()
	} else {
		m.Prepare()
	}
	c, lines := indraLines(b, dir, m.Cfg.Rows())
	b.Run("text", func(b *testing.B) {
		var s State
		px := 0
		for _, w := range c.Widths {
			px += w
		}
		b.ResetTimer()
		for range b.N {
			for i, w := range c.Widths {
				m.Forward(&s, lines[i], w)
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(px)*800/1e6, "ms/800px")
	})
	r := rand.New(rand.NewSource(1))
	for _, w := range []int{256, 800, 1600} {
		b.Run(fmt.Sprintf("w%d", w), func(b *testing.B) {
			ink := make([]float32, m.Cfg.Rows()*w)
			for i := range ink {
				if r.Float32() > 0.85 {
					ink[i] = 1
				}
			}
			var s State
			b.ResetTimer()
			for range b.N {
				m.Forward(&s, ink, w)
			}
		})
	}
}

// TestIndraInt8 reads the reference lines in int8: the same text.
func TestIndraInt8(t *testing.T) {
	dir := indraDir(t)
	f, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	q, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	f.Prepare()
	q.PrepareInt8()
	c, lines := indraLines(t, dir, f.Cfg.Rows())
	var sf, sq State
	for i, w := range c.Widths {
		f.Forward(&sf, lines[i], w)
		q.Forward(&sq, lines[i], w)
		a, _ := Greedy(sf.Logits, sf.L, f.Cfg.Vocab)
		b, _ := Greedy(sq.Logits, sq.L, q.Cfg.Vocab)
		if len(a) != len(b) {
			t.Errorf("line %d: %d symbols in float32, %d in int8", i, len(a), len(b))
			continue
		}
		for j := range a {
			if a[j].ID != b[j].ID {
				t.Errorf("line %d: symbol %d differs in int8", i, j)
			}
		}
	}
}
