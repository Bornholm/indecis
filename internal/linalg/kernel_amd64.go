package linalg

import "os"

// Micro-kernel in Go assembly for AVX2 + FMA, without cgo. Go 1.27's
// experimental SIMD generates a correct kernel but spills accumulators
// to the stack and multiplies register copies; the assembly keeps the
// whole tile in the 16 YMM registers.

//go:noescape
func microKernelAVX2(kb int, ap, bp, tile *float32)

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// avx2 reports whether the assembly kernel can be used: AVX2 and FMA
// present, YMM state enabled by the OS. INDECIS_NOASM=1 disables it, for
// comparison or as a workaround.
var avx2 = detectAVX2()

func detectAVX2() bool {
	if os.Getenv("INDECIS_NOASM") == "1" {
		return false
	}
	return cpuAVX2()
}

// cpuAVX2 reports whether the CPU and the OS support AVX2 and FMA.
func cpuAVX2() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	const fma, osxsave, avx = 1 << 12, 1 << 27, 1 << 28
	if ecx1&fma == 0 || ecx1&osxsave == 0 || ecx1&avx == 0 {
		return false
	}
	if xcr0, _ := xgetbv(); xcr0&6 != 6 { // XMM and YMM state saved
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0
}

// Assembly reports whether the assembly kernel is active.
func Assembly() bool { return avx2 }

func nr() int {
	if avx2 {
		return 16
	}
	return nrGo()
}

func microKernel(kb int, ap, bp, tile []float32, nr int) {
	if avx2 && nr == 16 {
		if kb == 0 {
			clear(tile[:mr*16])
			return
		}
		_ = ap[kb*mr-1]
		_ = bp[kb*16-1]
		_ = tile[mr*16-1]
		microKernelAVX2(kb, &ap[0], &bp[0], &tile[0])
		return
	}
	microKernelGo(kb, ap, bp, tile, nr)
}

//go:noescape
func microKernelVNNI(kq int, ap *uint8, lda int, bp *int8, tile *int32)

// vnni reports whether the int8 AVX-VNNI micro-kernel (VEX form) can be
// used. INDECIS_NOASM=1 disables it too.
var vnni = avx2 && detectVNNI()

func detectVNNI() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	eax71, _, _, _ := cpuid(7, 1)
	return eax71&(1<<4) != 0 // AVX-VNNI
}

// CPU reports what the processor supports, whether or not INDECIS_NOASM
// turned the assembly kernels off.
func CPU() (avx2FMA, avxVNNI bool) {
	avx2FMA = cpuAVX2()
	return avx2FMA, avx2FMA && detectVNNI()
}

// Int8Fast reports whether MatMul8 has a hardware kernel. Without it,
// the portable kernel is exact but much slower than MatMul.
func Int8Fast() bool { return vnni }

func microKernel8(kq int, ap []uint8, lda int, bp []int8, tile *[mr8 * nr8]int32) {
	if vnni {
		if kq == 0 {
			*tile = [mr8 * nr8]int32{}
			return
		}
		_ = ap[(mr8-1)*lda+kq*4-1]
		_ = bp[kq*nr8*4-1]
		microKernelVNNI(kq, &ap[0], lda, &bp[0], &tile[0])
		return
	}
	microKernel8Go(kq, ap, lda, bp, tile)
}

//go:noescape
func microKernelVNNIDeq(kq int, ap *uint8, lda int, bp *int8, c *float32, ldc int, sb, zc, sa *float32, accumulate int)

// microKernel8Deq computes a whole 6×16 tile of C, rows ldc apart:
// c[r][j] (+)= sa[r]·sb[j]·(Σ a·b − zc[j]), the rescaling done in the
// kernel's registers. It reports false, doing nothing, without the
// AVX-VNNI kernel: the caller then goes through microKernel8.
func microKernel8Deq(kq int, ap []uint8, lda int, bp []int8, c []float32, ldc int, sb, zc, sa []float32, accumulate bool) bool {
	if !vnni || kq == 0 {
		return false
	}
	_ = ap[(mr8-1)*lda+kq*4-1]
	_ = bp[kq*nr8*4-1]
	_ = c[(mr8-1)*ldc+nr8-1]
	_ = sb[nr8-1]
	_ = zc[nr8-1]
	_ = sa[mr8-1]
	acc := 0
	if accumulate {
		acc = 1
	}
	microKernelVNNIDeq(kq, &ap[0], lda, &bp[0], &c[0], ldc, &sb[0], &zc[0], &sa[0], acc)
	return true
}

//go:noescape
func quantRowAVX2(dst *int8, x, mean *float32, n int, flip int) (scale float32, sum int32)

//go:noescape
func quantColsAVX2(dst *int8, x, inv *float32, n int)

//go:noescape
func interleave4AVX2(dst, src *int8, n, stride int)

//go:noescape
func geluMulAVX2(o, a, b *float32, n int)

// geluMulFast computes GeluMul on the longest prefix of o that the AVX2
// kernel handles (a multiple of 16 values) and returns its length.
func geluMulFast(o, a, b []float32) int {
	n := len(o) &^ 15
	if !avx2 || n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	geluMulAVX2(&o[0], &a[0], &b[0], n)
	return n
}

// quantFast reports whether the AVX2 quantization of the attention
// operands applies to heads of D values.
func quantFast(D int) bool { return avx2 && D%32 == 0 }

// quantRow quantizes c = x − mean to int8, scale = max|c|/127 (see
// quantRowAVX2); flip gives uint8 with the 128 offset. len(x) = D,
// quantFast(D).
func quantRow(dst []int8, x, mean []float32, flip bool) (float32, int32) {
	n := len(x)
	_ = dst[n-1]
	_ = mean[n-1]
	f := 0
	if flip {
		f = 1
	}
	return quantRowAVX2(&dst[0], &x[0], &mean[0], n, f)
}

// quantCols writes dst[i] = round(x[i]·inv[i]), quantFast(len(x)).
func quantCols(dst []int8, x, inv []float32) {
	n := len(x)
	_ = dst[n-1]
	_ = inv[n-1]
	quantColsAVX2(&dst[0], &x[0], &inv[0], n)
}

// interleave4 writes the 4 rows of n bytes of src column by column, 16
// columns at dst[c·stride:] for each chunk c.
func interleave4(dst, src []int8, n, stride int) {
	_ = src[4*n-1]
	_ = dst[(n/16-1)*stride+63]
	interleave4AVX2(&dst[0], &src[0], n, stride)
}

func zeroUpper()

// ZeroUpper clears the upper halves of the AVX registers (VZEROUPPER).
// Go's portable SIMD code leaves them dirty, and every scalar float
// instruction (SSE, as Go compiles math.Exp) then pays a transition
// penalty: call it before a scalar-heavy loop that follows vector code.
func ZeroUpper() {
	if avx2 {
		zeroUpper()
	}
}
