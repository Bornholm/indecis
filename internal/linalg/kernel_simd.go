//go:build goexperiment.simd

package linalg

import "simd"

// Accelerated reports whether the SIMD kernels are compiled in.
const Accelerated = true

// nrGo returns the width of a tile: two vectors. Vector length is only
// known at runtime (128 bits in emulation or on Neon, 256 on AVX2, 512 on
// AVX-512), and does not change during execution.
//
// This is not a package variable: the simd doc notes that global
// initializers depending on SIMD do not work.
func nrGo() int {
	return 2 * simd.BroadcastFloat32s(0).Len()
}

// microKernelGo is never inlined: a SIMD function inlined into a closure
// crashes the compiler (Go 1.27, GOEXPERIMENT=simd).
//
// microKernelGo computes the tile mr×nr = sum_q ap[q]ᵀ·bp[q] and writes it
// to tile (row-major, stride nr). 12 accumulators + 2 B vectors + 1
// broadcast: 15 registers, which fits within AVX2's 16 ymm registers.
//
//go:noinline
func microKernelGo(kb int, ap, bp, tile []float32, nr int) {
	V := nr / 2
	var c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51 simd.Float32s
	for q := 0; q < kb; q++ {
		b := bp[q*nr : q*nr+nr]
		a := ap[q*mr : q*mr+mr]
		b0 := simd.LoadFloat32s(b)
		b1 := simd.LoadFloat32s(b[V:])
		x := simd.BroadcastFloat32s(a[0])
		c00 = x.MulAdd(b0, c00)
		c01 = x.MulAdd(b1, c01)
		x = simd.BroadcastFloat32s(a[1])
		c10 = x.MulAdd(b0, c10)
		c11 = x.MulAdd(b1, c11)
		x = simd.BroadcastFloat32s(a[2])
		c20 = x.MulAdd(b0, c20)
		c21 = x.MulAdd(b1, c21)
		x = simd.BroadcastFloat32s(a[3])
		c30 = x.MulAdd(b0, c30)
		c31 = x.MulAdd(b1, c31)
		x = simd.BroadcastFloat32s(a[4])
		c40 = x.MulAdd(b0, c40)
		c41 = x.MulAdd(b1, c41)
		x = simd.BroadcastFloat32s(a[5])
		c50 = x.MulAdd(b0, c50)
		c51 = x.MulAdd(b1, c51)
	}
	c00.Store(tile[0*nr:])
	c01.Store(tile[0*nr+V:])
	c10.Store(tile[1*nr:])
	c11.Store(tile[1*nr+V:])
	c20.Store(tile[2*nr:])
	c21.Store(tile[2*nr+V:])
	c30.Store(tile[3*nr:])
	c31.Store(tile[3*nr+V:])
	c40.Store(tile[4*nr:])
	c41.Store(tile[4*nr+V:])
	c50.Store(tile[5*nr:])
	c51.Store(tile[5*nr+V:])
}
