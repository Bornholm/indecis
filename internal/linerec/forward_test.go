package linerec

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const tinyDir = "../../testdata/ocr-tiny"

type tinyCases struct {
	Shape   []int   `json:"shape"`
	Widths  []int   `json:"widths"`
	Targets [][]int `json:"targets"`
	LR      float64 `json:"lr"`
	WD      float64 `json:"weight_decay"`
	Clip    float64 `json:"clip"`
	Layer   struct {
		Lengths   []int   `json:"lengths"`
		Loss      float64 `json:"loss"`
		TotalNorm float64 `json:"total_norm"`
	} `json:"layer"`
}

func readF32(t testing.TB, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func loadTiny(t *testing.T) (tinyCases, [][]float32) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(tinyDir, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c tinyCases
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	x := readF32(t, filepath.Join(tinyDir, "x.f32"))
	H, Wm := c.Shape[2], c.Shape[3]
	lines := make([][]float32, len(c.Widths))
	for i, w := range c.Widths {
		ink := make([]float32, H*w)
		for y := range H {
			copy(ink[y*w:(y+1)*w], x[(i*H+y)*Wm:])
		}
		lines[i] = ink
	}
	return c, lines
}

func maxDiff(a, b []float32) float64 {
	var d float64
	for i := range a {
		d = max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d
}

func TestForwardMatchesPyTorch(t *testing.T) {
	c, lines := loadTiny(t)
	for _, norm := range []string{"batch", "layer"} {
		t.Run(norm, func(t *testing.T) {
			m, err := Load(filepath.Join(tinyDir, norm))
			if err != nil {
				t.Fatal(err)
			}
			want := readF32(t, filepath.Join(tinyDir, norm, "logits.f32"))
			V := m.Cfg.Vocab
			var s State
			off := 0
			for i, w := range c.Widths {
				m.Forward(&s, lines[i], w)
				if s.L != c.Layer.Lengths[i] {
					t.Fatalf("line %d: %d steps, want %d", i, s.L, c.Layer.Lengths[i])
				}
				n := s.L * V
				if d := maxDiff(s.Logits[:n], want[off:off+n]); d > 1e-5 {
					t.Errorf("line %d: max |Δ logits| = %.2e", i, d)
				}
				off += n
			}
		})
	}
}

func TestGreedy(t *testing.T) {
	// Steps: a, a, blank, a, b, b → "a a b".
	V := 3
	best := []int{1, 1, 0, 1, 2, 2}
	logits := make([]float32, len(best)*V)
	for t, b := range best {
		logits[t*V+b] = 5
	}
	syms, conf := Greedy(logits, len(best), V)
	var ids []int
	for _, s := range syms {
		ids = append(ids, s.ID)
	}
	if len(ids) != 3 || ids[0] != 1 || ids[1] != 1 || ids[2] != 2 || syms[2].Steps != 2 {
		t.Fatalf("ids %v, syms %+v", ids, syms)
	}
	if conf < 0.97 || conf > 0.99 {
		t.Fatalf("confidence %v", conf)
	}
}
