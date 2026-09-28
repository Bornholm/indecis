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
