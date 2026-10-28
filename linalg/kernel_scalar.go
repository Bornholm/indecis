//go:build !goexperiment.simd

package linalg

// Accelerated reports whether the SIMD kernels are compiled in.
const Accelerated = false

func nrGo() int { return 8 }

// microKernelGo is the scalar version: same contract as the SIMD
// version, so that the rest of the package can be shared.
//
//go:noinline
func microKernelGo(kb int, ap, bp, tile []float32, nr int) {
	var acc [mr * 8]float32
	for q := 0; q < kb; q++ {
		b := bp[q*8 : q*8+8]
		a := ap[q*mr : q*mr+mr]
		for r, av := range a {
			row := acc[r*8 : r*8+8]
			for c, bv := range b {
				row[c] += av * bv
			}
		}
	}
	copy(tile, acc[:])
}

// SIMDInfo reports the vector width in bits and whether the portable SIMD
// runs in hardware: without SIMD compiled in, there are no vectors.
func SIMDInfo() (bits int, emulated bool) { return 0, false }
