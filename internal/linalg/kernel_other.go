//go:build !amd64

package linalg

// Assembly indique si le noyau assembleur est actif.
func Assembly() bool { return false }

func nr() int { return nrGo() }

func microKernel(kb int, ap, bp, tile []float32, nr int) {
	microKernelGo(kb, ap, bp, tile, nr)
}

// Int8Fast indique si MatMul8 dispose d'un noyau matériel.
func Int8Fast() bool { return false }

func microKernel8(kq int, ap []uint8, lda int, bp []int8, tile *[mr8 * nr8]int32) {
	microKernel8Go(kq, ap, lda, bp, tile)
}
