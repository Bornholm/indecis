package linalg

import (
	"math"
)

// Produit matriciel quantifié en int8, pour l'inférence.
//
// Les poids (B) sont quantifiés une fois, par colonne de op(B) (un canal de
// sortie) : b ≈ sb[j]·qb, qb ∈ [-127, 127]. Les activations (A) le sont à
// chaque appel, par ligne : a ≈ sa[i]·(qa − 128), qa ∈ [1, 255], le
// décalage de 128 rendant qa non signé comme l'exige VPDPBUSD. Alors
//
//	C[i][j] ≈ sa[i]·sb[j]·(Σ qa·qb − 128·Σ qb)
//
// et la somme entière est exacte : k ≤ 16 000 ne peut pas déborder un int32.
// Sur un encodeur de phrases, l'écart est de l'ordre de 1e-3 en relatif,
// sans effet mesurable sur les décisions (tools/infbench -int8 -eval).

const (
	mr8 = 6  // lignes d'une tuile
	nr8 = 16 // colonnes d'une tuile
)

// PackedB8 est un opérande B quantifié et empaqueté : panneaux de 16
// colonnes, chaque colonne portant 4 profondeurs consécutives par mot de 32
// bits.
type PackedB8 struct {
	k, n   int
	kq     int    // profondeur en quadruplets, k arrondi au-dessus
	data   []int8 // panneau p, quadruplet q, colonne c, octet t → [p·kq·64 + q·64 + c·4 + t]
	scale  []float32
	colsum []int32
}

// PackB8 quantifie et empaquette op(B), k×n : B est stockée k×n, ou n×k si
// transB.
func PackB8(b []float32, k, n int, transB bool) *PackedB8 {
	if len(b) < k*n {
		panic("linalg: PackB8: slice too short")
	}
	at := func(q, j int) float32 { // op(B)[q][j]
		if transB {
			return b[j*k+q]
		}
		return b[q*n+j]
	}
	kq := (k + 3) / 4
	panels := (n + nr8 - 1) / nr8
	p := &PackedB8{k: k, n: n, kq: kq,
		data:   make([]int8, panels*kq*nr8*4),
		scale:  make([]float32, n),
		colsum: make([]int32, n),
	}
	for j := 0; j < n; j++ {
		var mx float32
		for q := 0; q < k; q++ {
			mx = max(mx, abs32(at(q, j)))
		}
		if mx == 0 {
			continue
		}
		s := mx / 127
		p.scale[j] = s
		base := (j/nr8)*kq*nr8*4 + (j%nr8)*4
		for q := 0; q < k; q++ {
			v := int8(math.Round(float64(at(q, j) / s)))
			p.data[base+(q/4)*nr8*4+q%4] = v
			p.colsum[j] += int32(v)
		}
	}
	return p
}

// Size est le nombre d'octets occupés.
func (p *PackedB8) Size() int { return len(p.data) + 8*p.n }

// MatMul8 calcule C = A·op(B) (ou C += si accumulate), A étant m×k, avec la
// quantification int8 décrite plus haut.
func MatMul8(c, a []float32, b *PackedB8, m int, accumulate bool) {
	k, n := b.k, b.n
	if m == 0 || n == 0 {
		return
	}
	if len(c) < m*n || len(a) < m*k {
		panic("linalg: MatMul8: slice too short")
	}
	kq := b.kq
	rowPanels := (m + mr8 - 1) / mr8
	apack := getBytes(rowPanels * kq * mr8 * 4)
	defer putBytes(apack)
	sa := getBuf(rowPanels * mr8)
	defer putBuf(sa)
	quantizeA(apack, sa, a, m, k, kq)

	colPanels := (n + nr8 - 1) / nr8
	work := func(lo, hi int) {
		var tile [mr8 * nr8]int32
		for pj := lo; pj < hi; pj++ {
			bp := b.data[pj*kq*nr8*4 : (pj+1)*kq*nr8*4]
			j0 := pj * nr8
			cols := min(nr8, n-j0)
			for pi := 0; pi < rowPanels; pi++ {
				ap := apack[pi*kq*mr8*4 : (pi+1)*kq*mr8*4]
				microKernel8(kq, ap, bp, &tile)
				rows := min(mr8, m-pi*mr8)
				for r := 0; r < rows; r++ {
					i := pi*mr8 + r
					s := sa[i]
					dst := c[i*n+j0 : i*n+j0+cols]
					acc := tile[r*nr8 : r*nr8+cols]
					for x := range dst {
						v := s * b.scale[j0+x] * float32(acc[x]-128*b.colsum[j0+x])
						if accumulate {
							dst[x] += v
						} else {
							dst[x] = v
						}
					}
				}
			}
		}
	}
	// Même règle que MatMul : un worker par tranche de calcul suffisante.
	active := min(Workers(), max(1, m*n*k/(4*macsPerWorker)))
	if active == 1 {
		work(0, colPanels)
		return
	}
	Parallel(colPanels, max(1, colPanels/(4*active)), work)
}

// quantizeA quantifie chaque ligne de A et l'empaquette par panneaux de 6
// lignes : panneau p, quadruplet q, ligne r, octet t → [p·kq·24 + q·24 +
// r·4 + t]. Le zéro quantifié vaut 128 ; les profondeurs et lignes de
// complément aussi.
func quantizeA(apack []uint8, sa, a []float32, m, k, kq int) {
	for i := range apack {
		apack[i] = 128
	}
	for i := 0; i < m; i++ {
		row := a[i*k : (i+1)*k]
		var mx float32
		for _, v := range row {
			mx = max(mx, abs32(v))
		}
		sa[i] = 0
		if mx == 0 {
			continue
		}
		s := mx / 127
		sa[i] = s
		inv := 1 / s
		dst := apack[(i/mr8)*kq*mr8*4+(i%mr8)*4:]
		for q, v := range row {
			dst[(q/4)*mr8*4+q%4] = uint8(int32(roundHalfEven(v*inv)) + 128)
		}
	}
}

func abs32(v float32) float32 { return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)) }

// roundHalfEven arrondit au plus proche, à égalité vers le pair, pour
// |v| < 2²² : ajouter puis retrancher 1,5·2²³ fait arrondir le matériel.
func roundHalfEven(v float32) float32 {
	const magic = 12582912
	return (v + magic) - magic
}

// microKernel8Go est la référence portable du micro-noyau int8.
func microKernel8Go(kq int, ap []uint8, bp []int8, tile *[mr8 * nr8]int32) {
	*tile = [mr8 * nr8]int32{}
	for q := 0; q < kq; q++ {
		a := ap[q*mr8*4 : (q+1)*mr8*4]
		b := bp[q*nr8*4 : (q+1)*nr8*4]
		for r := 0; r < mr8; r++ {
			for c := 0; c < nr8; c++ {
				var s int32
				for t := 0; t < 4; t++ {
					s += int32(a[r*4+t]) * int32(b[c*4+t])
				}
				tile[r*nr8+c] += s
			}
		}
	}
}

var bytePool = make(chan []uint8, 16)

func getBytes(n int) []uint8 {
	select {
	case b := <-bytePool:
		if cap(b) >= n {
			return b[:n]
		}
	default:
	}
	return make([]uint8, n)
}

func putBytes(b []uint8) {
	select {
	case bytePool <- b:
	default:
	}
}
