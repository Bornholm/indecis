package linalg

import (
	"math"
)

// int8-quantized matrix product, for inference.
//
// Weights (B) are quantized once, per column of op(B) (an output
// channel): b ~ sb[j]·qb, qb in [-127, 127]. Activations (A) are
// quantized on each call, per row: a ~ sa[i]·(qa - 128), qa in [1, 255],
// the 128 offset making qa unsigned as VPDPBUSD requires. Then
//
//	C[i][j] ~ sa[i]·sb[j]·(sum qa·qb - 128·sum qb)
//
// and the integer sum is exact: k <= 16,000 cannot overflow an int32. On
// a sentence encoder, the gap is on the order of 1e-3 relative, with no
// measurable effect on decisions (tools/infbench -int8 -eval).

const (
	mr8 = 6  // rows of a tile
	nr8 = 16 // columns of a tile
)

// PackedB8 is a quantized and packed B operand: panels of 16 columns,
// each column carrying 4 consecutive depths per 32-bit word.
type PackedB8 struct {
	k, n   int
	kq     int    // depth in quadruplets, k rounded up
	data   []int8 // panel p, quadruplet q, column c, byte t -> [p·kq·64 + q·64 + c·4 + t]
	scale  []float32
	colsum []int32
	zc     []float32 // 128·colsum, subtracted from the integer sums
}

// PackB8 quantizes and packs op(B), k×n: B is stored k×n, or n×k if
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
	p.zc = make([]float32, n)
	for j, c := range p.colsum {
		p.zc[j] = 128 * float32(c)
	}
	return p
}

// Size is the number of bytes used.
func (p *PackedB8) Size() int { return len(p.data) + 8*p.n }

// MatMul8 computes C = A·op(B) (or C += if accumulate), A being m×k,
// using the int8 quantization described above.
func MatMul8(c, a []float32, b *PackedB8, m int, accumulate bool) {
	MatMul8N(c, a, b, m, accumulate, Workers())
}

// MatMul8N is MatMul8 with at most limit workers.
func MatMul8N(c, a []float32, b *PackedB8, m int, accumulate bool, limit int) {
	k, n := b.k, b.n
	if m == 0 || n == 0 {
		return
	}
	if len(c) < m*n || len(a) < m*k {
		panic("linalg: MatMul8: slice too short")
	}
	kq := b.kq
	lda := kq * 4
	rowPanels := (m + mr8 - 1) / mr8
	apack := getBytes(rowPanels * mr8 * lda)
	defer putBytes(apack)
	sa := getBuf(rowPanels * mr8)
	defer putBuf(sa)
	quantizeA(apack, sa, a, m, k, lda)

	colPanels := (n + nr8 - 1) / nr8
	// Each worker handles a slice of column panels [lo, hi) for all row
	// panels: the tiles of a row panel accumulate into acc, then each row
	// is rescaled in one go (one vector call per row, not per tile).
	work := func(lo, hi int) {
		var tile [mr8 * nr8]int32
		j0, j1 := lo*nr8, min(n, hi*nr8)
		width := j1 - j0
		acc := getInts(mr8 * width)
		defer putInts(acc)
		for pi := 0; pi < rowPanels; pi++ {
			ap := apack[pi*mr8*lda : (pi+1)*mr8*lda]
			for pj := lo; pj < hi; pj++ {
				microKernel8(kq, ap, lda, b.data[pj*kq*nr8*4:(pj+1)*kq*nr8*4], &tile)
				c0 := pj*nr8 - j0
				cols := min(nr8, n-pj*nr8)
				for r := 0; r < mr8; r++ {
					copy(acc[r*width+c0:r*width+c0+cols], tile[r*nr8:r*nr8+cols])
				}
			}
			rows := min(mr8, m-pi*mr8)
			for r := 0; r < rows; r++ {
				i := pi*mr8 + r
				Dequantize(c[i*n+j0:i*n+j1], acc[r*width:(r+1)*width], b.scale[j0:j1], b.zc[j0:j1], sa[i], accumulate)
			}
		}
	}
	// Same rule as MatMul: one worker per sufficient slice of work.
	active := min(max(1, min(limit, Workers())), max(1, m*n*k/(4*macsPerWorker)))
	if active == 1 {
		work(0, colPanels)
		return
	}
	ParallelN(active, colPanels, max(1, colPanels/(4*active)), work)
}

// quantizeA quantizes each row of A to uint8 (quantized zero: 128), one
// row of lda bytes per row of A. Padding depths and rows beyond m are
// set to 128.
func quantizeA(apack []uint8, sa, a []float32, m, k, lda int) {
	for i := 0; i < len(apack)/lda; i++ {
		row := apack[i*lda : (i+1)*lda]
		sa[i] = 0
		if i >= m {
			fill128(row)
			continue
		}
		x := a[i*k : (i+1)*k]
		mx := AbsMax(x)
		if mx == 0 {
			fill128(row)
			continue
		}
		s := mx / 127
		sa[i] = s
		QuantizeRow(row[:k], x, 1/s)
		fill128(row[k:])
	}
}

func fill128(b []uint8) {
	for i := range b {
		b[i] = 128
	}
}

func abs32(v float32) float32 { return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)) }

// roundHalfEven rounds to nearest, ties to even, for |v| < 2^22: adding
// then subtracting 1.5·2^23 makes the hardware do the rounding.
func roundHalfEven(v float32) float32 {
	const magic = 12582912
	return (v + magic) - magic
}

// microKernel8Go is the portable reference for the int8 micro-kernel: 6
// rows of A spaced lda bytes apart, a B panel of 16 columns.
func microKernel8Go(kq int, ap []uint8, lda int, bp []int8, tile *[mr8 * nr8]int32) {
	*tile = [mr8 * nr8]int32{}
	for r := 0; r < mr8; r++ {
		arow := ap[r*lda:]
		for q := 0; q < kq; q++ {
			a := arow[q*4 : q*4+4]
			b := bp[q*nr8*4 : (q+1)*nr8*4]
			for c := 0; c < nr8; c++ {
				tile[r*nr8+c] += int32(a[0])*int32(b[c*4]) + int32(a[1])*int32(b[c*4+1]) +
					int32(a[2])*int32(b[c*4+2]) + int32(a[3])*int32(b[c*4+3])
			}
		}
	}
}

var intPool = make(chan []int32, 16)

func getInts(n int) []int32 {
	select {
	case b := <-intPool:
		if cap(b) >= n {
			return b[:n]
		}
	default:
	}
	return make([]int32, n)
}

func putInts(b []int32) {
	select {
	case intPool <- b:
	default:
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

// maxOutliers bounds the input channels MatMul8Outliers takes out of the
// int8 product.
const maxOutliers = 32

// MatMul8Outliers is MatMul8 for activations with a few massive input
// channels, as vision transformers have (the LLM.int8() decomposition):
// the columns of A whose maximum magnitude exceeds ratio times the mean
// of the column maxima, 32 at most, are left out of the int8 product, so
// they no longer flatten the per-row quantization, and multiplied in
// float32 by the dequantized weights of those rows of op(B). The columns
// are chosen on each call, from A itself: no calibration.
func MatMul8Outliers(c, a []float32, b *PackedB8, m int, accumulate bool, ratio float32) {
	k, n := b.k, b.n
	if m == 0 || n == 0 {
		return
	}
	colMax := getBuf(k)
	defer putBuf(colMax)
	clear(colMax)
	for i := 0; i < m; i++ {
		for j, v := range a[i*k : (i+1)*k] {
			colMax[j] = max(colMax[j], abs32(v))
		}
	}
	var mean float32
	for _, v := range colMax {
		mean += v
	}
	mean /= float32(k)
	var out []int
	for j, v := range colMax {
		if v > ratio*mean {
			out = append(out, j)
		}
	}
	if len(out) == 0 {
		MatMul8(c, a, b, m, accumulate)
		return
	}
	if len(out) > maxOutliers {
		// Keep the largest.
		for i := 0; i < maxOutliers; i++ {
			best := i
			for x := i + 1; x < len(out); x++ {
				if colMax[out[x]] > colMax[out[best]] {
					best = x
				}
			}
			out[i], out[best] = out[best], out[i]
		}
		out = out[:maxOutliers]
	}
	rest := getBuf(m * k)
	defer putBuf(rest)
	copy(rest, a[:m*k])
	for i := 0; i < m; i++ {
		for _, j := range out {
			rest[i*k+j] = 0
		}
	}
	MatMul8(c, rest, b, m, accumulate)
	// Rows j of op(B), dequantized: data[p·kq·64 + q·64 + col·4 + t] holds
	// op(B)[4q+t][16p+col].
	w := getBuf(n)
	defer putBuf(w)
	for _, j := range out {
		q, t := j/4, j%4
		for col := 0; col < n; col++ {
			v := b.data[(col/nr8)*b.kq*nr8*4+q*nr8*4+(col%nr8)*4+t]
			w[col] = float32(v) * b.scale[col]
		}
		for i := 0; i < m; i++ {
			x := a[i*k+j]
			ci := c[i*n : (i+1)*n]
			for col, wv := range w[:n] {
				ci[col] += x * wv
			}
		}
	}
}
