package linalg

import (
	"os"
	"runtime"
	_ "unsafe" // go:linkname
)

// Micro-kernels in Go assembly for arm64 (NEON), without cgo: the float32
// kernel uses FMLA, present on every arm64 processor; the int8 kernel
// UDOT, from the ARMv8.2 dotprod extension (Cortex-A55/A75 and later,
// almost every phone since 2018). Without dotprod, MatMul8 runs the
// portable kernel and Int8Fast reports false.

//go:noescape
func microKernelNEON(kb int, ap, bp, tile *float32)

// neon reports whether the assembly kernels can be used. INDECIS_NOASM=1
// disables them, for comparison or as a workaround.
var neon = os.Getenv("INDECIS_NOASM") != "1"

// Assembly reports whether the assembly kernel is active.
func Assembly() bool { return neon }

func nr() int {
	if neon {
		return 16
	}
	return nrGo()
}

func microKernel(kb int, ap, bp, tile []float32, nr int) {
	if neon && nr == 16 {
		if kb == 0 {
			clear(tile[:mr*16])
			return
		}
		_ = ap[kb*mr-1]
		_ = bp[kb*16-1]
		_ = tile[mr*16-1]
		microKernelNEON(kb, &ap[0], &bp[0], &tile[0])
		return
	}
	microKernelGo(kb, ap, bp, tile, nr)
}

//go:noescape
func microKernelDot(kq int, ap *uint8, lda int, bp *int8, tile *int32)

//go:noescape
func microKernelDotDeq(kq int, ap *uint8, lda int, bp *int8, c *float32, ldc int, sb, zc, sa *float32, accumulate int)

// dot reports whether the int8 UDOT micro-kernel can be used.
var dot = neon && cpuDotProd()

// cpuDotProd reports whether the processor has the dotprod instructions
// (SDOT, UDOT): on Linux and Android from the HWCAP the kernel passes in
// the auxiliary vector, always on arm64 macOS (Apple M1 and later), never
// elsewhere (iOS devices before the A13 lack it).
func cpuDotProd() bool {
	switch runtime.GOOS {
	case "darwin":
		return true
	case "linux", "android":
		return hwcap()&hwcapASIMDDP != 0
	}
	return false
}

const (
	atHWCAP      = 16      // AT_HWCAP
	hwcapASIMDDP = 1 << 20 // HWCAP_ASIMDDP
)

// runtimeAuxv returns the auxiliary vector the kernel passed to the
// process, read by the runtime at startup (pairs tag, value), as
// golang.org/x/sys/cpu does.
//
//go:linkname runtimeAuxv runtime.getAuxv
func runtimeAuxv() []uintptr

// hwcap returns AT_HWCAP, 0 if the auxiliary vector lacks it.
func hwcap() uint64 {
	auxv := runtimeAuxv()
	for i := 0; i+1 < len(auxv); i += 2 {
		if auxv[i] == atHWCAP {
			return uint64(auxv[i+1])
		}
	}
	return 0
}

// CPU reports what the processor supports. Both are x86 features: false
// on arm64 (Int8Fast reports the dotprod kernel).
func CPU() (avx2FMA, avxVNNI bool) { return false, false }

// Int8Fast reports whether MatMul8 has a hardware kernel. Without it,
// the portable kernel is exact but much slower than MatMul.
func Int8Fast() bool { return dot }

func microKernel8(kq int, ap []uint8, lda int, bp []int8, tile *[mr8 * nr8]int32) {
	if dot {
		if kq == 0 {
			*tile = [mr8 * nr8]int32{}
			return
		}
		_ = ap[(mr8-1)*lda+kq*4-1]
		_ = bp[kq*nr8*4-1]
		microKernelDot(kq, &ap[0], lda, &bp[0], &tile[0])
		return
	}
	microKernel8Go(kq, ap, lda, bp, tile)
}

// microKernel8Deq computes a whole 6×16 tile of C, rows ldc apart:
// c[r][j] (+)= sa[r]·sb[j]·(Σ a·b − zc[j]), the rescaling done in the
// kernel's registers. It reports false, doing nothing, without the
// dotprod kernel: the caller then goes through microKernel8.
func microKernel8Deq(kq int, ap []uint8, lda int, bp []int8, c []float32, ldc int, sb, zc, sa []float32, accumulate bool) bool {
	if !dot || kq == 0 {
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
	microKernelDotDeq(kq, &ap[0], lda, &bp[0], &c[0], ldc, &sb[0], &zc[0], &sa[0], acc)
	return true
}

// The attention quantization and GeluMul have no arm64 kernel: the Go
// code runs.

func quantFast(D int) bool { return false }

func geluMulFast(o, a, b []float32) int { return 0 }

func quantRow(dst []int8, x, mean []float32, flip bool) (float32, int32) {
	panic("linalg: no arm64 quantization")
}

func quantCols(dst []int8, x, inv []float32) { panic("linalg: no arm64 quantization") }

func interleave4(dst, src []int8, n, stride int) { panic("linalg: no arm64 quantization") }

// ZeroUpper does nothing on arm64 (see the amd64 version).
func ZeroUpper() {}
