package indecis

import (
	"runtime"

	"github.com/bornholm/indecis/internal/linalg"
)

// Hardware describes what speeds up computation on the current machine.
type Hardware struct {
	// Arch is the processor architecture (runtime.GOARCH).
	Arch string
	// SIMD reports whether the binary was built with Go's portable SIMD
	// (GOEXPERIMENT=simd).
	SIMD bool
	// SIMDEmulated reports whether that SIMD is emulated in pure Go because
	// the processor lacks the instructions: correct, but slow.
	SIMDEmulated bool
	// VectorBits is the SIMD vector width: 128 (SSE, Neon), 256 (AVX2) or
	// 512 (AVX-512); 0 without SIMD.
	VectorBits int
	// AVX2FMA reports whether the processor supports AVX2 and FMA, used by
	// the float32 assembly matrix kernel (amd64).
	AVX2FMA bool
	// AVXVNNI reports whether the processor supports AVX-VNNI, used by the
	// int8 matrix kernel (WithInt8).
	AVXVNNI bool
	// Assembly reports whether the assembly kernels are active: supported
	// by the processor and not turned off by INDECIS_NOASM=1.
	Assembly bool
}

// DetectHardware reports the acceleration available on this machine.
func DetectHardware() Hardware {
	h := Hardware{Arch: runtime.GOARCH, SIMD: linalg.Accelerated, Assembly: linalg.Assembly()}
	h.VectorBits, h.SIMDEmulated = linalg.SIMDInfo()
	h.AVX2FMA, h.AVXVNNI = linalg.CPU()
	return h
}
