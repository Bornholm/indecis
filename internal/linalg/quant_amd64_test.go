package linalg

import (
	"math/rand/v2"
	"testing"
)

// The AVX2 quantization gives the values of the Go code: same scale, same
// rounding, same order.
func TestQuantAVX2(t *testing.T) {
	if !quantFast(32) {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{32, 64, 128} {
		for trial := 0; trial < 50; trial++ {
			x, mean, inv := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range x {
				x[i] = float32(r.NormFloat64() * 3)
				mean[i] = float32(r.NormFloat64())
				inv[i] = float32(10 + r.Float64()*20)
				if trial == 0 {
					x[i] = mean[i] // all zero once centered
				}
			}
			c := make([]float32, n)
			for i := range c {
				c[i] = x[i] - mean[i]
			}
			mx := AbsMax(c)
			for _, flip := range []bool{false, true} {
				dst := make([]int8, n)
				scale, sum := quantRow(dst, x, mean, flip)
				if mx == 0 {
					if scale != 0 || sum != 0 {
						t.Fatalf("n=%d zero row: scale %v sum %d", n, scale, sum)
					}
					continue
				}
				s := mx / 127
				if scale != s {
					t.Fatalf("n=%d scale %v, want %v", n, scale, s)
				}
				var want int32
				for i, v := range c {
					q := int8(roundHalfEven(v * (1 / s)))
					want += int32(q)
					w := q
					if flip {
						w = int8(uint8(q) ^ 0x80)
					}
					if dst[i] != w {
						t.Fatalf("n=%d flip=%v dst[%d] = %d, want %d", n, flip, i, dst[i], w)
					}
				}
				if sum != want {
					t.Fatalf("n=%d sum %d, want %d", n, sum, want)
				}
			}
			if trial == 0 {
				continue
			}
			for i := range inv {
				inv[i] = 127 / (mx + float32(r.Float64()))
			}
			dst := make([]int8, n)
			quantCols(dst, c, inv)
			for i, v := range c {
				if w := int8(roundHalfEven(v * inv[i])); dst[i] != w {
					t.Fatalf("n=%d quantCols[%d] = %d, want %d", n, i, dst[i], w)
				}
			}
		}
		src := make([]int8, 4*n)
		for i := range src {
			src[i] = int8(r.IntN(256) - 128)
		}
		stride := 100
		dst := make([]int8, (n/16-1)*stride+64)
		interleave4(dst, src, n, stride)
		for j := 0; j < n; j++ {
			for t4 := 0; t4 < 4; t4++ {
				if got := dst[(j/16)*stride+(j%16)*4+t4]; got != src[t4*n+j] {
					t.Fatalf("n=%d interleave4 col %d row %d = %d, want %d", n, j, t4, got, src[t4*n+j])
				}
			}
		}
	}
}
