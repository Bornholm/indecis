package linalg

import (
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"
)

// attendRef is the attention in float64: softmax(q·kᵀ·scale)·v, keys
// limited to |i − j| <= window when window >= 0.
func attendRef(q, k, v []float32, n, D, window int, scale float64) []float64 {
	o := make([]float64, n*D)
	s := make([]float64, n)
	for i := 0; i < n; i++ {
		mx := math.Inf(-1)
		for j := 0; j < n; j++ {
			s[j] = math.Inf(-1)
			if window >= 0 && (j < i-window || j > i+window) {
				continue
			}
			var dot float64
			for d := 0; d < D; d++ {
				dot += float64(q[i*D+d]) * float64(k[j*D+d])
			}
			s[j] = dot * scale
			mx = max(mx, s[j])
		}
		var sum float64
		for j := range s {
			s[j] = math.Exp(s[j] - mx)
			sum += s[j]
		}
		for j := 0; j < n; j++ {
			for d := 0; d < D; d++ {
				o[i*D+d] += s[j] / sum * float64(v[j*D+d])
			}
		}
	}
	return o
}

// Attend8 stays close to the float64 attention, global or local, at
// lengths that are not multiples of the panels, with keys that share a
// large offset (what the centering of K removes). The scores spread over
// about 4 (q and k of standard deviation 2): a sharp softmax, where the
// quantization of the scores shows most. On a trained model, the masks
// are those of float32 attention (TAB, indecis-ner).
func TestAttend8(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	var b AttnBuffers
	for _, D := range []int{64, 32} {
		for _, n := range []int{1, 5, 17, 48, 100, 259} {
			for _, window := range []int{-1, 8, 64} {
				q, k, v := make([]float32, n*D), make([]float32, n*D), make([]float32, n*D)
				for i := range q {
					q[i] = float32(r.NormFloat64()) * 2
					k[i] = float32(r.NormFloat64())*2 + float32(i%D%7) // offset per dimension
					v[i] = float32(r.NormFloat64())
				}
				scale := 1 / math.Sqrt(float64(D))
				want := attendRef(q, k, v, n, D, window, scale)
				o := make([]float32, n*D)
				for i := range o {
					o[i] = float32(math.NaN()) // every value must be written
				}
				Attend8(o, q, k, v, n, D, window, float32(scale), &b)
				var num, den, worst float64
				for i := range o {
					d := float64(o[i]) - want[i]
					num += d * d
					den += want[i] * want[i]
					worst = max(worst, math.Abs(d))
				}
				rel := math.Sqrt(num / max(den, 1e-30))
				if math.IsNaN(rel) || rel > 0.03 || worst > 0.25 {
					t.Errorf("D=%d n=%d window=%d: relative error %.4f, worst %.4f", D, n, window, rel, worst)
				}
			}
		}
	}
}

// Ranges of queries computed apart, into a wider output, give exactly the
// whole attention.
func TestAttn8Ranges(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	n, D, ldo := 259, 64, 64*3
	q, k, v := make([]float32, n*D), make([]float32, n*D), make([]float32, n*D)
	for i := range q {
		q[i], k[i], v[i] = float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64())
	}
	for _, window := range []int{-1, 64} {
		var b AttnBuffers
		want := make([]float32, n*D)
		Attend8(want, q, k, v, n, D, window, 0.125, &b)
		var a Attn8
		a.Prepare(q, k, v, n, D, 0.125)
		o := make([]float32, n*ldo)
		for i0 := 0; i0 < n; i0 += AttnBlock {
			var sc AttnScratch
			i1 := min(n, i0+AttnBlock)
			a.Attend(o[i0*ldo+D:], ldo, i0, i1, window, &sc)
		}
		for i := 0; i < n; i++ {
			for d := 0; d < D; d++ {
				if o[i*ldo+D+d] != want[i*D+d] {
					t.Fatalf("window=%d: o[%d][%d] = %v, want %v", window, i, d, o[i*ldo+D+d], want[i*D+d])
				}
			}
		}
	}
}

func BenchmarkAttend8(b *testing.B) {
	n, D := benchN(), 64
	r := rand.New(rand.NewPCG(1, 2))
	q, k, v := make([]float32, n*D), make([]float32, n*D), make([]float32, n*D)
	for i := range q {
		q[i], k[i], v[i] = float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64())
	}
	o := make([]float32, n*D)
	var buf AttnBuffers
	for b.Loop() {
		Attend8(o, q, k, v, n, D, -1, 0.125, &buf)
	}
}

func benchN() int {
	if n, err := strconv.Atoi(os.Getenv("ATTN_N")); err == nil {
		return n
	}
	return 258
}
