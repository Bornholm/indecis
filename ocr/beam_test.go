package ocr

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/bornholm/indecis/internal/charlm"
	"github.com/bornholm/indecis/internal/linerec"
)

// BenchmarkReadLinesLM is BenchmarkReadLines with a language model
// (INDECIS_OCR_BENCH_LM), every line through the beam search.
func BenchmarkReadLinesLM(b *testing.B) {
	dir, data, lmPath := os.Getenv("INDECIS_OCR_BENCH_MODEL"), os.Getenv("INDECIS_OCR_BENCH_DATA"), os.Getenv("INDECIS_OCR_BENCH_LM")
	if dir == "" || data == "" || lmPath == "" {
		b.Skip("INDECIS_OCR_BENCH_MODEL, _DATA and _LM not set")
	}
	lm, err := charlm.Load(lmPath)
	if err != nil {
		b.Fatal(err)
	}
	o := DefaultLM()
	o.Gate = 2
	r, err := Load(dir, WithInt8(), WithLanguageModel(lm, o))
	if err != nil {
		b.Fatal(err)
	}
	lines, err := LoadLines(data, r.CropOptions())
	if err != nil {
		b.Fatal(err)
	}
	lines = lines[:min(len(lines), 200)]
	b.ResetTimer()
	for range b.N {
		for _, l := range lines {
			r.ReadCrop(Crop{Ink: l.Ink, W: l.W})
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(len(lines))/1e6, "ms/line")
}

// TestBeamDecode: without the language model the search reads the best
// path; with it, a step torn between two letters goes the language's way.
func TestBeamDecode(t *testing.T) {
	chars := []string{"c", "h", "a", "t", "n"}
	V := len(chars) + 1
	id := func(s string) int {
		for i, c := range chars {
			if c == s {
				return i + 1
			}
		}
		return 0
	}
	// "c h ? t": the third step hesitates between a (0.45) and n (0.55).
	steps := []map[int]float32{{id("c"): 0.99}, {0: 0.99}, {id("h"): 0.99}, {id("n"): 0.55, id("a"): 0.45}, {id("t"): 0.99}}
	logits := make([]float32, len(steps)*V)
	for t, s := range steps {
		for k := range V {
			p, ok := s[k]
			if !ok {
				p = 0.001
			}
			logits[t*V+k] = float32(math.Log(float64(p)))
		}
	}
	b := charlm.NewBuilder(3)
	for range 50 {
		b.Add("chat chat le chat")
	}
	lm := b.Build(1)
	text := func(syms []linerec.Symbol) string {
		var s strings.Builder
		for _, sy := range syms {
			s.WriteString(chars[sy.ID-1])
		}
		return s.String()
	}
	o := DefaultLM()
	o.Alpha, o.Beta = 0, 0
	if got := text(beamDecode(logits, len(steps), V, chars, lm, o)); got != "chnt" {
		t.Fatalf("without the model: %q", got)
	}
	o = DefaultLM()
	if got := text(beamDecode(logits, len(steps), V, chars, lm, o)); got != "chat" {
		t.Fatalf("with the model: %q", got)
	}
}
