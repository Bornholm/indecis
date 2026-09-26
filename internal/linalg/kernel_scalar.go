//go:build !goexperiment.simd

package linalg

// Accelerated indique si les noyaux SIMD sont compilés.
const Accelerated = false

func nrGo() int { return 8 }

// microKernelGo est la version scalaire : même contrat que la version SIMD,
// pour que tout le reste du package soit partagé.
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
