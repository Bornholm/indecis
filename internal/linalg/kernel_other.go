//go:build !amd64

package linalg

// Assembly indique si le noyau assembleur est actif.
func Assembly() bool { return false }

func nr() int { return nrGo() }

func microKernel(kb int, ap, bp, tile []float32, nr int) {
	microKernelGo(kb, ap, bp, tile, nr)
}
