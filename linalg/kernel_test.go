package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Checks of the architecture's micro-kernels (AVX2 and AVX-VNNI on amd64,
// NEON and dotprod on arm64, the Go code elsewhere) against references
// written plainly, on depths that are not multiples of the unrolling and
// rows of A that are not aligned.

// The float32 kernel computes the tile mr×nr = Σ_q ap[q]ᵀ·bp[q] and
// overwrites it, whatever the depth.
func TestMicroKernelMatchesReference(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	nr := nr()
	for _, kb := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 63, 255, 256} {
		ap := randSlice(r, kb*mr+1)[1:] // not 16-byte aligned
		bp := randSlice(r, kb*nr)
		tile := make([]float32, mr*nr)
		for i := range tile {
			tile[i] = float32(math.NaN())
		}
		microKernel(kb, ap, bp, tile, nr)
		for i := 0; i < mr; i++ {
			for j := 0; j < nr; j++ {
				var want, mag float64
				for q := 0; q < kb; q++ {
					p := float64(ap[q*mr+i]) * float64(bp[q*nr+j])
					want += p
					mag += math.Abs(p)
				}
				got := float64(tile[i*nr+j])
				if math.IsNaN(got) || math.Abs(got-want) > 1e-6*mag+1e-30 {
					t.Fatalf("kb=%d [%d][%d]: %v, want %v", kb, i, j, got, want)
				}
			}
		}
	}
}

// operands8 fills A (6 rows lda apart) and B (a packed panel) by kind:
// random, or the extremes of the ranges.
func operands8(r *rand.Rand, kind string, kq, lda int) ([]uint8, []int8) {
	ap := make([]uint8, 5*lda+kq*4+1)[1:] // last row exactly as long as needed
	bp := make([]int8, kq*nr8*4)
	for i := range ap {
		switch kind {
		case "max":
			ap[i] = 255
		case "mixed":
			ap[i] = uint8(255 * (i % 2))
		default:
			ap[i] = uint8(r.Intn(256))
		}
	}
	for i := range bp {
		switch kind {
		case "max":
			bp[i] = 127
		case "mixed":
			bp[i] = int8(-127 + 254*((i/3)%2))
		default:
			bp[i] = int8(r.Intn(255) - 127)
		}
	}
	return ap, bp
}

var depths8 = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 31, 96, 289}

// The int8 kernel gives exactly the integers of microKernel8Go.
func TestMicroKernel8Edges(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for _, kind := range []string{"random", "max", "mixed"} {
		for _, kq := range depths8 {
			for _, pad := range []int{0, 3, 12} {
				lda := kq*4 + pad
				ap, bp := operands8(r, kind, kq, lda)
				var got, want [mr8 * nr8]int32
				for i := range got {
					got[i] = -1
				}
				microKernel8(kq, ap, lda, bp, &got)
				microKernel8Go(kq, ap, lda, bp, &want)
				if got != want {
					t.Fatalf("%s kq=%d lda=%d: %v\nwant %v", kind, kq, lda, got[:8], want[:8])
				}
			}
		}
	}
}

// The kernel's own rescaling gives, bit for bit, what Dequantize gives
// from the integer tile: (acc − zc)·sb·sa, then added to C.
func TestMicroKernel8DeqMatchesDequantize(t *testing.T) {
	if !Int8Fast() {
		t.Skip("no int8 kernel")
	}
	r := rand.New(rand.NewSource(6))
	for _, kq := range depths8[1:] {
		for _, ldc := range []int{nr8, 23} {
			for _, accumulate := range []bool{false, true} {
				name := fmt.Sprintf("kq=%d/ldc=%d/acc=%v", kq, ldc, accumulate)
				lda := kq*4 + 5
				ap, bp := operands8(r, "random", kq, lda)
				sb, zc, sa := randSlice(r, nr8), randSlice(r, nr8), randSlice(r, mr8)
				for j := range zc {
					zc[j] *= 1e5
				}
				c := randSlice(r, (mr8-1)*ldc+nr8)
				want := append([]float32(nil), c...)
				if !microKernel8Deq(kq, ap, lda, bp, c, ldc, sb, zc, sa, accumulate) {
					t.Fatalf("%s: microKernel8Deq declined", name)
				}
				var tile [mr8 * nr8]int32
				microKernel8Go(kq, ap, lda, bp, &tile)
				for i := 0; i < mr8; i++ {
					for j := 0; j < nr8; j++ {
						// Explicit conversions: no fused multiply-add.
						v := float32(float32(float32(float32(tile[i*nr8+j])-zc[j])*sb[j]) * sa[i])
						if accumulate {
							v = want[i*ldc+j] + v
						}
						want[i*ldc+j] = v
					}
				}
				for i := range c {
					if math.Float32bits(c[i]) != math.Float32bits(want[i]) {
						t.Fatalf("%s: [%d] %v, want %v", name, i, c[i], want[i])
					}
				}
			}
		}
	}
}
