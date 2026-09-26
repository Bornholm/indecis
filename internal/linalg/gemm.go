// Package linalg fournit les noyaux numériques de l'entraînement et de
// l'inférence : produit matriciel, opérations vectorielles.
//
// Tout est en float32, en row-major contigu. Le produit matriciel suit le
// schéma de GotoBLAS : les deux opérandes sont recopiés (« packés ») en
// panneaux contigus, puis un micro-noyau calcule des tuiles mr×nr en
// registres. Le packing absorbe les transpositions : les trois formes dont la
// rétropropagation a besoin (A·B, A·Bᵀ, Aᵀ·B) partagent le même micro-noyau.
//
// Seul le micro-noyau dépend de l'architecture. Il existe en deux versions :
// SIMD portable (GOEXPERIMENT=simd) et scalaire. Le reste du package est
// commun.
//
// Le code qui mène au SIMD n'utilise pas de closures : avec le support
// expérimental de Go 1.27, une closure appelant une fonction SIMD fait
// planter le compilateur.
package linalg

import (
	"sync"
	"sync/atomic"
)

const (
	mr = 6   // lignes d'une tuile du micro-noyau
	kc = 256 // profondeur d'un bloc de k : un panneau de B tient en L1/L2
	mc = 72  // lignes de A par tâche (multiple de mr)
	nc = 256 // colonnes de C par tâche (arrondi à un multiple de nr)

	// Multiplications-additions minimales par worker (~0,2 ms en AVX2).
	macsPerWorker = 8 << 20
)

// MatMul calcule C = op(A)·op(B), ou C += op(A)·op(B) si accumulate.
//
// op(A) est m×k : A est stockée m×k, ou k×m si transA.
// op(B) est k×n : B est stockée k×n, ou n×k si transB.
// C est m×n.
func MatMul(c, a, b []float32, m, k, n int, transA, transB, accumulate bool) {
	if m == 0 || n == 0 {
		return
	}
	if len(c) < m*n || len(a) < m*k || len(b) < k*n {
		panic("linalg: MatMul: slice too short")
	}
	if k == 0 {
		if !accumulate {
			clear(c[:m*n])
		}
		return
	}

	g := &gemm{
		c: c, a: a, b: b,
		m: m, k: k, n: n,
		transA: transA, transB: transB,
		accumulate: accumulate,
		nr:         nr(),
	}
	g.run(Workers())
}

// MatMulSerial est MatMul sans parallélisme interne, pour les appels faits
// depuis des tâches déjà parallèles (une tête d'attention par goroutine).
func MatMulSerial(c, a, b []float32, m, k, n int, transA, transB, accumulate bool) {
	if m == 0 || n == 0 {
		return
	}
	if len(c) < m*n || len(a) < m*k || len(b) < k*n {
		panic("linalg: MatMulSerial: slice too short")
	}
	if k == 0 {
		if !accumulate {
			clear(c[:m*n])
		}
		return
	}
	g := &gemm{
		c: c, a: a, b: b,
		m: m, k: k, n: n,
		transA: transA, transB: transB,
		accumulate: accumulate,
		nr:         nr(),
	}
	g.run(1)
}

type gemm struct {
	c, a, b        []float32
	m, k, n        int
	transA, transB bool
	accumulate     bool
	nr             int
	ncr            int // nc arrondi à un multiple de nr

	// bloc de k courant
	pc, kb int
	bpack  []float32
	// pre, s'il est défini, fournit B déjà empaquetée.
	pre *PackedB

	tasksM, tasksN int
	next           atomic.Int64
}

func (g *gemm) run(limit int) {
	g.ncr = (nc + g.nr - 1) / g.nr * g.nr
	g.tasksM = (g.m + mc - 1) / mc
	// Autant de workers que le calcul en justifie : lancer et synchroniser
	// une goroutine coûte plus que quelques millions de multiplications.
	// Une phrase seule (28 tokens) tourne ainsi deux fois plus vite sur un
	// cœur que répartie sur quatorze.
	active := min(limit, max(1, g.m*g.n*g.k/macsPerWorker))
	// Peu de lignes : on découpe plus finement les colonnes pour occuper
	// les workers retenus.
	if g.tasksM < active && active > 1 {
		// Quatre tâches par worker : sur un processeur hybride, les cœurs
		// lents (E, LP-E) n'en retiennent qu'une petite part.
		want := (4*active + g.tasksM - 1) / g.tasksM
		cols := (g.n + want - 1) / want
		g.ncr = max(g.nr, (cols+g.nr-1)/g.nr*g.nr)
	}
	g.tasksN = (g.n + g.ncr - 1) / g.ncr

	if g.pre == nil {
		panels := (g.n + g.nr - 1) / g.nr
		g.bpack = getBuf(panels * g.nr * min(kc, g.k))
		defer putBuf(g.bpack)
	}

	workers := min(active, g.tasksM*g.tasksN)

	for g.pc = 0; g.pc < g.k; g.pc += kc {
		g.kb = min(kc, g.k-g.pc)
		if g.pre != nil {
			g.bpack = g.pre.blocks[g.pc/kc]
		} else {
			g.packB()
		}
		g.next.Store(0)
		if workers == 1 {
			g.work()
			continue
		}
		var wg sync.WaitGroup
		wg.Add(workers)
		for range workers {
			go g.worker(&wg)
		}
		wg.Wait()
	}
}

func (g *gemm) worker(wg *sync.WaitGroup) {
	defer wg.Done()
	g.work()
}

// work traite des tâches (bloc de lignes × bloc de colonnes) jusqu'à épuisement.
func (g *gemm) work() {
	apack := getBuf(mc * g.kb)
	tile := getBuf(mr * g.nr)
	defer putBuf(apack)
	defer putBuf(tile)

	total := int64(g.tasksM * g.tasksN)
	for {
		t := g.next.Add(1) - 1
		if t >= total {
			return
		}
		ti, tj := int(t)/g.tasksN, int(t)%g.tasksN
		i0, i1 := ti*mc, min(ti*mc+mc, g.m)
		j0, j1 := tj*g.ncr, min(tj*g.ncr+g.ncr, g.n)
		g.packA(apack, i0, i1)
		g.compute(apack, tile, i0, i1, j0, j1)
	}
}

// compute enchaîne les micro-noyaux sur le bloc [i0,i1)×[j0,j1).
func (g *gemm) compute(apack, tile []float32, i0, i1, j0, j1 int) {
	nr, kb := g.nr, g.kb
	store := g.pc == 0 && !g.accumulate
	for j := j0; j < j1; j += nr {
		bp := g.bpack[(j/nr)*nr*kb : (j/nr+1)*nr*kb]
		cols := min(nr, j1-j)
		for i := i0; i < i1; i += mr {
			ap := apack[((i-i0)/mr)*mr*kb : ((i-i0)/mr+1)*mr*kb]
			microKernel(kb, ap, bp, tile, nr)
			rows := min(mr, i1-i)
			for r := 0; r < rows; r++ {
				dst := g.c[(i+r)*g.n+j : (i+r)*g.n+j+cols]
				src := tile[r*nr : r*nr+cols]
				if store {
					copy(dst, src)
				} else {
					for x, v := range src {
						dst[x] += v
					}
				}
			}
		}
	}
}

// packA recopie les lignes [i0,i1) × colonnes [pc,pc+kb) de op(A) en
// panneaux de mr lignes : panneau p, profondeur q → apack[p·mr·kb + q·mr + r].
// Les lignes au-delà de m sont complétées par des zéros.
func (g *gemm) packA(apack []float32, i0, i1 int) {
	kb, pc := g.kb, g.pc
	for i := i0; i < i1; i += mr {
		dst := apack[((i-i0)/mr)*mr*kb : ((i-i0)/mr+1)*mr*kb]
		rows := min(mr, i1-i)
		if g.transA {
			// A stockée k×m : op(A)[i][q] = A[q][i]
			for q := 0; q < kb; q++ {
				src := g.a[(pc+q)*g.m+i : (pc+q)*g.m+i+rows]
				d := dst[q*mr : q*mr+mr]
				copy(d, src)
				clear(d[rows:])
			}
			continue
		}
		for r := 0; r < mr; r++ {
			if r >= rows {
				for q := 0; q < kb; q++ {
					dst[q*mr+r] = 0
				}
				continue
			}
			src := g.a[(i+r)*g.k+pc : (i+r)*g.k+pc+kb]
			for q, v := range src {
				dst[q*mr+r] = v
			}
		}
	}
}

// packB recopie les lignes [pc,pc+kb) de op(B), toutes colonnes, en panneaux
// de nr colonnes : panneau p, profondeur q → bpack[p·nr·kb + q·nr + c].
func (g *gemm) packB() {
	nr, kb, pc := g.nr, g.kb, g.pc
	for j := 0; j < g.n; j += nr {
		dst := g.bpack[(j/nr)*nr*kb : (j/nr+1)*nr*kb]
		cols := min(nr, g.n-j)
		if g.transB {
			// B stockée n×k : op(B)[q][j] = B[j][q]
			for c := 0; c < nr; c++ {
				if c >= cols {
					for q := 0; q < kb; q++ {
						dst[q*nr+c] = 0
					}
					continue
				}
				src := g.b[(j+c)*g.k+pc : (j+c)*g.k+pc+kb]
				for q, v := range src {
					dst[q*nr+c] = v
				}
			}
			continue
		}
		for q := 0; q < kb; q++ {
			src := g.b[(pc+q)*g.n+j : (pc+q)*g.n+j+cols]
			d := dst[q*nr : q*nr+nr]
			copy(d, src)
			clear(d[cols:])
		}
	}
}

// PackedB est un opérande B empaqueté une fois pour toutes. Les poids d'un
// modèle ne changent pas entre deux requêtes : les empaqueter à chaque
// produit coûtait près d'un cinquième du temps d'inférence.
type PackedB struct {
	k, n, nr int
	blocks   [][]float32 // un bloc par tranche de kc lignes de op(B)
}

// PackB empaquette op(B), k×n : B est stockée k×n, ou n×k si transB. Le
// résultat ne dépend plus de b, qui peut changer ensuite.
func PackB(b []float32, k, n int, transB bool) *PackedB {
	if len(b) < k*n {
		panic("linalg: PackB: slice too short")
	}
	p := &PackedB{k: k, n: n, nr: nr()}
	g := &gemm{b: b, k: k, n: n, transB: transB, nr: p.nr}
	panels := (n + p.nr - 1) / p.nr
	for g.pc = 0; g.pc < k; g.pc += kc {
		g.kb = min(kc, k-g.pc)
		g.bpack = make([]float32, panels*p.nr*g.kb)
		g.packB()
		p.blocks = append(p.blocks, g.bpack)
	}
	return p
}

// Size est le nombre de float32 occupés.
func (p *PackedB) Size() int {
	n := 0
	for _, b := range p.blocks {
		n += len(b)
	}
	return n
}

// MatMulPacked est MatMul avec un B empaqueté par PackB : C = A·op(B), A
// étant m×k.
func MatMulPacked(c, a []float32, b *PackedB, m int, accumulate bool) {
	k, n := b.k, b.n
	if m == 0 || n == 0 || k == 0 {
		if !accumulate && k == 0 {
			clear(c[:m*n])
		}
		return
	}
	if len(c) < m*n || len(a) < m*k {
		panic("linalg: MatMulPacked: slice too short")
	}
	if b.nr != nr() {
		panic("linalg: MatMulPacked: B empaquetée pour un autre micro-noyau")
	}
	g := &gemm{
		c: c, a: a,
		m: m, k: k, n: n,
		accumulate: accumulate,
		nr:         b.nr,
		pre:        b,
	}
	g.run(Workers())
}

var bufPool sync.Pool

func getBuf(n int) []float32 {
	if p, ok := bufPool.Get().(*[]float32); ok && cap(*p) >= n {
		return (*p)[:n]
	}
	return make([]float32, n)
}

func putBuf(b []float32) {
	bufPool.Put(&b)
}
