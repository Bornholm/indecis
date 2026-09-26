//go:build goexperiment.simd

package linalg

import "simd"

// Accelerated indique si les noyaux SIMD sont compilés.
const Accelerated = true

// nrGo retourne la largeur d'une tuile : deux vecteurs. La longueur des
// vecteurs n'est connue qu'à l'exécution (128 bits en émulation ou sur Neon,
// 256 en AVX2, 512 en AVX-512), et ne change pas pendant l'exécution.
//
// Ce n'est pas une variable de package : la doc de simd signale que les
// initialiseurs globaux dépendant du SIMD ne fonctionnent pas.
func nrGo() int {
	return 2 * simd.BroadcastFloat32s(0).Len()
}

// microKernelGo n'est jamais inliné : une fonction SIMD inlinée dans une
// closure fait planter le compilateur (Go 1.27, GOEXPERIMENT=simd).
//
// microKernelGo calcule la tuile mr×nr = Σ_q ap[q]ᵀ·bp[q] et l'écrit dans tile
// (row-major, pas nr). 12 accumulateurs + 2 vecteurs de B + 1 broadcast :
// 15 registres, ce que permettent les 16 registres ymm d'AVX2.
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
