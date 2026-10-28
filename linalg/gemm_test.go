package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func randSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2 - 1
	}
	return s
}

// naive computes the reference in float64.
func naive(a, b []float32, m, k, n int, transA, transB bool) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for q := 0; q < k; q++ {
				av := a[i*k+q]
				if transA {
					av = a[q*m+i]
				}
				bv := b[q*n+j]
				if transB {
					bv = b[j*k+q]
				}
				s += float64(av) * float64(bv)
			}
			out[i*n+j] = s
		}
	}
	return out
}

func TestMatMul_AllLayoutsAndEdges(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	// Sizes chosen to hit the edges: m not a multiple of mr,
	// n not a multiple of nr, k crossing a kc block, and large enough
	// to trigger parallelism.
	shapes := [][3]int{
		{1, 1, 1}, {5, 3, 7}, {6, 16, 16}, {7, 17, 33},
		{13, 300, 29}, {73, 257, 259}, {150, 520, 300},
	}
	for _, sh := range shapes {
		m, k, n := sh[0], sh[1], sh[2]
		for _, tA := range []bool{false, true} {
			for _, tB := range []bool{false, true} {
				name := fmt.Sprintf("%dx%dx%d/tA=%v/tB=%v", m, k, n, tA, tB)
				t.Run(name, func(t *testing.T) {
					a, b := randSlice(r, m*k), randSlice(r, k*n)
					want := naive(a, b, m, k, n, tA, tB)

					c := randSlice(r, m*n) // garbage values: must be overwritten
					MatMul(c, a, b, m, k, n, tA, tB, false)
					check(t, c, want, nil, k)

					init := randSlice(r, m*n)
					acc := append([]float32(nil), init...)
					MatMul(acc, a, b, m, k, n, tA, tB, true)
					check(t, acc, want, init, k)
				})
			}
		}
	}
}

func check(t *testing.T, got []float32, want []float64, init []float32, k int) {
	t.Helper()
	// float32 rounding error: proportional to sqrt(k) for terms on the
	// order of 1.
	tol := 1e-5 * math.Sqrt(float64(k)) * 4
	for i := range want {
		w := want[i]
		if init != nil {
			w += float64(init[i])
		}
		if d := math.Abs(float64(got[i]) - w); d > tol {
			t.Fatalf("index %d: got %v want %v (|d| = %g > %g)", i, got[i], w, d, tol)
		}
	}
}

func TestMatMul_ZeroK(t *testing.T) {
	c := []float32{1, 2, 3, 4}
	MatMul(c, nil, nil, 2, 0, 2, false, false, false)
	for _, v := range c {
		if v != 0 {
			t.Fatalf("got %v", c)
		}
	}
	c = []float32{1, 2, 3, 4}
	MatMul(c, nil, nil, 2, 0, 2, false, false, true)
	if c[3] != 4 {
		t.Fatalf("accumulate with k=0 must leave C: %v", c)
	}
}

// Shapes taken from bekko-a8m: batch of 16 × 96 tokens, dimension 384,
// MLP 2×1152.
func BenchmarkMatMul(b *testing.B) {
	cases := []struct {
		name           string
		m, k, n        int
		transA, transB bool
	}{
		{"fwd_Wi_x·Wᵀ", 1536, 384, 2304, false, true},
		{"bwd_dX_dY·W", 1536, 2304, 384, false, false},
		{"bwd_dW_dYᵀ·X", 2304, 1536, 384, true, false},
		{"attn_QKᵀ", 96, 64, 96, false, true},
	}
	r := rand.New(rand.NewSource(1))
	for _, cs := range cases {
		b.Run(cs.name, func(b *testing.B) {
			a, bb := randSlice(r, cs.m*cs.k), randSlice(r, cs.k*cs.n)
			c := make([]float32, cs.m*cs.n)
			b.ResetTimer()
			for range b.N {
				MatMul(c, a, bb, cs.m, cs.k, cs.n, cs.transA, cs.transB, false)
			}
			flops := 2 * float64(cs.m*cs.k*cs.n) * float64(b.N)
			b.ReportMetric(flops/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func TestMatMulPacked_MatchesMatMul(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, s := range [][3]int{{1, 1, 1}, {7, 384, 1152}, {15, 1152, 384}, {33, 300, 17}, {80, 520, 40}} {
		m, k, n := s[0], s[1], s[2]
		for _, transB := range []bool{false, true} {
			a := randSlice(r, m*k)
			b := randSlice(r, k*n)
			want := make([]float32, m*n)
			MatMul(want, a, b, m, k, n, false, transB, false)
			p := PackB(b, k, n, transB)
			orig := append([]float32(nil), b...)
			clear(b) // the pack no longer depends on b
			p.Unpack(b, transB)
			for i := range b {
				if b[i] != orig[i] {
					t.Fatalf("%v transB=%v: Unpack [%d] %v, want %v", s, transB, i, b[i], orig[i])
				}
			}
			got := make([]float32, m*n)
			MatMulPacked(got, a, p, m, false)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%v transB=%v: [%d] %v, want %v", s, transB, i, got[i], want[i])
				}
			}
			MatMulPacked(got, a, p, m, true)
			for i := range got {
				if got[i] != 2*want[i] && math.Abs(float64(got[i]-2*want[i])) > 1e-4 {
					t.Fatalf("%v accumulate: [%d] %v, want %v", s, i, got[i], 2*want[i])
				}
			}
		}
	}
}
