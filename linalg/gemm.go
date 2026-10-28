// Package linalg provides the numerical kernels for training and
// inference: matrix product, vector operations.
//
// Everything is float32, contiguous row-major. The matrix product follows
// the GotoBLAS scheme: both operands are copied ("packed") into contiguous
// panels, then a micro-kernel computes mr×nr tiles in registers. Packing
// absorbs the transpositions: the three forms backpropagation needs
// (A·B, A·Bᵀ, Aᵀ·B) share the same micro-kernel.
//
// Only the micro-kernel is architecture-dependent. It exists in two
// versions: portable SIMD (GOEXPERIMENT=simd) and scalar. The rest of the
// package is shared.
//
// The code leading to SIMD does not use closures: with Go 1.27's
// experimental support, a closure calling a SIMD function crashes the
// compiler.
package linalg

import (
	"sync"
	"sync/atomic"
)

const (
	mr = 6   // rows of a micro-kernel tile
	kc = 256 // depth of a k block: a B panel fits in L1/L2
	mc = 72  // rows of A per task (multiple of mr)
	nc = 256 // columns of C per task (rounded to a multiple of nr)

	// Minimum multiply-adds per worker (~0.2 ms on AVX2).
	macsPerWorker = 8 << 20
)

// MatMul computes C = op(A)·op(B), or C += op(A)·op(B) if accumulate.
//
// op(A) is m×k: A is stored m×k, or k×m if transA.
// op(B) is k×n: B is stored k×n, or n×k if transB.
// C is m×n.
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

// MatMulSerial is MatMul without internal parallelism, for calls made from
// tasks that are already parallel (one attention head per goroutine).
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
	ncr            int // nc rounded to a multiple of nr

	// current k block
	pc, kb int
	bpack  []float32
	// pre, if set, provides B already packed.
	pre *PackedB

	tasksM, tasksN int
	next           atomic.Int64
}

func (g *gemm) run(limit int) {
	g.ncr = (nc + g.nr - 1) / g.nr * g.nr
	g.tasksM = (g.m + mc - 1) / mc
	// As many workers as the computation justifies: launching and
	// synchronizing a goroutine costs more than a few million
	// multiplications. A single sentence (28 tokens) thus runs twice as
	// fast on one core as spread over fourteen.
	active := min(limit, max(1, g.m*g.n*g.k/macsPerWorker))
	// Few rows: split the columns more finely to keep the chosen workers
	// busy.
	if g.tasksM < active && active > 1 {
		// Four tasks per worker: on a hybrid processor, the slow cores
		// (E, LP-E) only pick up a small share.
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

// work processes tasks (row block × column block) until exhausted.
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

// compute chains the micro-kernels over the block [i0,i1)×[j0,j1).
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

// packA copies rows [i0,i1) × columns [pc,pc+kb) of op(A) into panels of
// mr rows: panel p, depth q -> apack[p·mr·kb + q·mr + r].
// Rows beyond m are padded with zeros.
func (g *gemm) packA(apack []float32, i0, i1 int) {
	kb, pc := g.kb, g.pc
	for i := i0; i < i1; i += mr {
		dst := apack[((i-i0)/mr)*mr*kb : ((i-i0)/mr+1)*mr*kb]
		rows := min(mr, i1-i)
		if g.transA {
			// A stored k×m: op(A)[i][q] = A[q][i]
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

// packB copies rows [pc,pc+kb) of op(B), all columns, into panels of nr
// columns: panel p, depth q -> bpack[p·nr·kb + q·nr + c].
func (g *gemm) packB() {
	nr, kb, pc := g.nr, g.kb, g.pc
	for j := 0; j < g.n; j += nr {
		dst := g.bpack[(j/nr)*nr*kb : (j/nr+1)*nr*kb]
		cols := min(nr, g.n-j)
		if g.transB {
			// B stored n×k: op(B)[q][j] = B[j][q]
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

// PackedB is a B operand packed once and for all. A model's weights don't
// change between requests: packing them on every product used to cost
// nearly a fifth of inference time.
type PackedB struct {
	k, n, nr int
	blocks   [][]float32 // one block per kc-row slice of op(B)
}

// PackB packs op(B), k×n: B is stored k×n, or n×k if transB. The result no
// longer depends on b, which can change afterwards.
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

// Unpack rewrites B, in the layout given to PackB, from the packed form.
func (p *PackedB) Unpack(b []float32, transB bool) {
	k, n, nr := p.k, p.n, p.nr
	if len(b) < k*n {
		panic("linalg: Unpack: slice too short")
	}
	for bi, blk := range p.blocks {
		pc := bi * kc
		kb := min(kc, k-pc)
		for j := 0; j < n; j++ {
			panel := blk[(j/nr)*nr*kb:]
			c := j % nr
			for q := 0; q < kb; q++ {
				v := panel[q*nr+c]
				if transB {
					b[j*k+pc+q] = v
				} else {
					b[(pc+q)*n+j] = v
				}
			}
		}
	}
}

// Size is the number of float32 elements used.
func (p *PackedB) Size() int {
	n := 0
	for _, b := range p.blocks {
		n += len(b)
	}
	return n
}

// MatMulPacked is MatMul with a B packed by PackB: C = A·op(B), A being
// m×k.
func MatMulPacked(c, a []float32, b *PackedB, m int, accumulate bool) {
	MatMulPackedN(c, a, b, m, accumulate, Workers())
}

// MatMulPackedN is MatMulPacked with at most limit workers.
func MatMulPackedN(c, a []float32, b *PackedB, m int, accumulate bool, limit int) {
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
		panic("linalg: MatMulPacked: B packed for a different micro-kernel")
	}
	g := &gemm{
		c: c, a: a,
		m: m, k: k, n: n,
		accumulate: accumulate,
		nr:         b.nr,
		pre:        b,
	}
	g.run(max(1, min(limit, Workers())))
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
