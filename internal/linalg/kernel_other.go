//go:build !amd64

package linalg

// Assembly reports whether the assembly kernel is active.
func Assembly() bool { return false }

func nr() int { return nrGo() }

func microKernel(kb int, ap, bp, tile []float32, nr int) {
	microKernelGo(kb, ap, bp, tile, nr)
}

// Int8Fast reports whether MatMul8 has a hardware kernel.
func Int8Fast() bool { return false }

func microKernel8(kq int, ap []uint8, lda int, bp []int8, tile *[mr8 * nr8]int32) {
	microKernel8Go(kq, ap, lda, bp, tile)
}

// CPU reports what the processor supports: the assembly kernels exist on
// amd64 only.
func CPU() (avx2FMA, avxVNNI bool) { return false, false }

func microKernel8Deq(kq int, ap []uint8, lda int, bp []int8, c []float32, ldc int, sb, zc, sa []float32, accumulate bool) bool {
	return false
}

func quantFast(D int) bool { return false }

func geluMulFast(o, a, b []float32) int { return 0 }

func quantRow(dst []int8, x, mean []float32, flip bool) (float32, int32) {
	panic("linalg: no AVX2 quantization")
}

func quantCols(dst []int8, x, inv []float32) { panic("linalg: no AVX2 quantization") }

func interleave4(dst, src []int8, n, stride int) { panic("linalg: no AVX2 quantization") }
