package linalg

import "os"

// Micro-noyau en assembleur Go pour AVX2 + FMA, sans cgo. Le SIMD
// expérimental de Go 1.27 génère un noyau correct mais qui renvoie des
// accumulateurs vers la pile et multiplie les copies de registres ;
// l'assembleur tient la tuile entière dans les 16 registres YMM.

//go:noescape
func microKernelAVX2(kb int, ap, bp, tile *float32)

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// avx2 indique si le noyau assembleur est utilisable : AVX2 et FMA
// présents, état YMM activé par le système. INDECIS_NOASM=1 le désactive,
// pour comparer ou contourner.
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
	if xcr0, _ := xgetbv(); xcr0&6 != 6 { // états XMM et YMM sauvegardés
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0
}

// Assembly indique si le noyau assembleur est actif.
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
