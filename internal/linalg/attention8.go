package linalg

import "unsafe"

// Attention in int8, for inference: both products of the attention, Q·Kᵀ
// and P·V, run on the int8 micro-kernel of MatMul8.
//
//   - Q is quantized per row (a query), like the activations of MatMul8.
//   - K is quantized per row (a key), after its mean over the sequence is
//     subtracted: q·(k − k̄) differs from q·k by a constant per query,
//     which the softmax ignores, and the centered keys quantize better
//     (the K smoothing of SageAttention, Zhang et al., ICLR 2025).
//   - P, the softmax weights, are in [0, 1] with a maximum of 1 per row:
//     they are quantized to uint8 [0, 255] without the 128 offset, so with
//     8 bits and no correction term. Each row is normalized by the sum of
//     its quantized weights.
//   - V is quantized per column (a dimension of the head).
//
// The softmax is computed on whole rows, a block of queries at a time:
// the keys a block sees are scored at once, so the running max and sum
// of attendBlocks are not needed. Memory stays linear in the length.

// AttnBlock is the number of queries scored at a time: a multiple of
// mr8. Attn8.Attend takes ranges of queries that start on a multiple of
// it.
const AttnBlock = 8 * mr8

// Attn8 holds the operands of an attention, quantized and packed once by
// Prepare: the ranges of queries can then be computed separately, by
// several workers (Attend).
type Attn8 struct {
	n, D, kq, lda int
	nPad, dPanels int
	qa            []uint8   // quantized Q, rows of lda bytes
	sq            []float32 // scale of each row of Q (times the score scale)
	kp            []int8    // packed centered K: panels of 16 keys, depth D
	ks, kzc       []float32 // scale and 128·sum of each key
	vp            []int8    // packed V: panels of 16 dimensions, depth the keys
	vs            []float32 // scale of each dimension of V
	zeros         []float32
	mean          []float32
	kc            []float32 // a centered key
	kb            []uint8   // a quantized key, or four quantized values
	vinv          []float32
}

// AttnScratch holds the buffers of Attn8.Attend, one per worker.
type AttnScratch struct {
	s  []float32 // scores of a block of queries
	pa []uint8   // quantized P of a block of queries
	sp []float32 // scale of each row of P (1/sum of its quantized weights)
}

// AttnBuffers holds the buffers of Attend8, reused from call to call.
type AttnBuffers struct {
	Attn8
	AttnScratch
}

func grow[T any](b []T, n int) []T {
	if cap(b) < n {
		return make([]T, n)
	}
	return b[:n]
}

// Attend8 writes into o (n×D) the attention of q over k, v (n×D each),
// with scores scaled by scale. window < 0: global attention; otherwise
// |i − j| ≤ window. q, k and v are not modified.
func Attend8(o, q, k, v []float32, n, D, window int, scale float32, b *AttnBuffers) {
	b.Prepare(q, k, v, D, n, D, scale)
	b.Attend(o, D, 0, n, window, &b.AttnScratch)
}

// Prepare quantizes and packs q, k and v, n rows of D values each, row j
// at [j·ld:], with scores scaled by scale. q, k and v are not modified
// nor kept.
func (b *Attn8) Prepare(q, k, v []float32, ld, n, D int, scale float32) {
	b.n, b.D = n, D
	if n == 0 {
		return
	}
	kq := (D + 3) / 4
	lda := kq * 4
	rowsPad := (n + mr8 - 1) / mr8 * mr8
	keyPanels := (n + nr8 - 1) / nr8
	nPad := keyPanels * nr8 // keys, padded to whole panels
	kqn := nPad / 4         // depth of P·V in quadruplets
	dPanels := (D + nr8 - 1) / nr8
	Dp := dPanels * nr8
	b.kq, b.lda, b.nPad, b.dPanels = kq, lda, nPad, dPanels
	b.zeros = grow(b.zeros, Dp)
	clear(b.zeros)
	if quantFast(D) {
		b.prepareAVX2(q, k, v, ld, n, D, scale)
		return
	}

	// Q, quantized per row; the score scale goes into the row scales.
	b.qa = grow(b.qa, rowsPad*lda)
	b.sq = grow(b.sq, rowsPad)
	for i := 0; i < rowsPad; i++ {
		row := b.qa[i*lda : (i+1)*lda]
		b.sq[i] = 0
		if i >= n {
			fill128(row)
			continue
		}
		x := q[i*ld : i*ld+D]
		mx := AbsMax(x)
		if mx == 0 {
			fill128(row)
			continue
		}
		sc := mx / 127
		b.sq[i] = sc * scale
		QuantizeRow(row[:D], x, 1/sc)
		fill128(row[D:])
	}

	// K, centered then quantized per key. A key quantized with the 128
	// offset becomes int8 by flipping the top bit of each byte; its
	// quadruplets of depths go to the panel as 4-byte words.
	b.mean = grow(b.mean, D)
	clear(b.mean)
	for j := 0; j < n; j++ {
		AddTo(b.mean, k[j*ld:j*ld+D])
	}
	Scale(b.mean, 1/float32(n))
	b.kp = grow(b.kp, keyPanels*kq*nr8*4)
	clear(b.kp)
	b.ks = grow(b.ks, nPad)
	b.kzc = grow(b.kzc, nPad)
	clear(b.ks)
	clear(b.kzc)
	b.kc = grow(b.kc, D)
	b.kb = grow(b.kb, 4*lda)
	for j := 0; j < n; j++ {
		kc := b.kc
		SubTo(kc, k[j*ld:j*ld+D], b.mean)
		mx := AbsMax(kc)
		if mx == 0 {
			continue
		}
		sc := mx / 127
		b.ks[j] = sc
		kb := b.kb[:lda]
		QuantizeRow(kb[:D], kc, 1/sc)
		fill128(kb[D:])
		dst := b.kp[(j/nr8)*kq*nr8*4+(j%nr8)*4:]
		var sum int32
		for d4 := 0; d4 < kq; d4++ {
			w := dst[d4*nr8*4 : d4*nr8*4+4]
			for t, u := range kb[d4*4 : d4*4+4] {
				w[t] = int8(u ^ 0x80)
				sum += int32(u)
			}
		}
		b.kzc[j] = 128 * float32(sum-128*int32(lda))
	}

	// V, quantized per dimension, its depth (the keys) padded with zeros:
	// four keys at a time, whose values of a dimension are a 4-byte word
	// of the panel.
	b.vp = grow(b.vp, dPanels*kqn*nr8*4)
	clear(b.vp)
	b.vs = grow(b.vs, Dp)
	clear(b.vs)
	for j := 0; j < n; j++ {
		MaxAbsTo(b.vs[:D], v[j*ld:j*ld+D])
	}
	b.vinv = grow(b.vinv, D)
	for d := 0; d < D; d++ {
		b.vinv[d] = 0
		if b.vs[d] > 0 {
			b.vs[d] /= 127
			b.vinv[d] = 1 / b.vs[d]
		}
	}
	vb := b.kb[:4*D]
	for j0 := 0; j0 < n; j0 += 4 {
		for t := 0; t < 4; t++ {
			if j0+t < n {
				QuantizeCols(vb[t*D:(t+1)*D], v[(j0+t)*ld:(j0+t)*ld+D], b.vinv)
			} else {
				fill128(vb[t*D : (t+1)*D])
			}
		}
		base := b.vp[(j0/4)*nr8*4:]
		for d := 0; d < D; d++ {
			w := base[(d/nr8)*kqn*nr8*4+(d%nr8)*4:]
			w = w[:4]
			w[0] = int8(vb[d] ^ 0x80)
			w[1] = int8(vb[D+d] ^ 0x80)
			w[2] = int8(vb[2*D+d] ^ 0x80)
			w[3] = int8(vb[3*D+d] ^ 0x80)
		}
	}
}

// prepareAVX2 is Prepare with the AVX2 quantization (quantFast(D), so
// lda = D): one call per row centers, scales, rounds and narrows it.
func (b *Attn8) prepareAVX2(q, k, v []float32, ld, n, D int, scale float32) {
	kq, nPad, dPanels := b.kq, b.nPad, b.dPanels
	rowsPad := (n + mr8 - 1) / mr8 * mr8
	kqn := nPad / 4

	b.qa = grow(b.qa, rowsPad*D)
	b.sq = grow(b.sq, rowsPad)
	qa := asInt8(b.qa)
	for i := 0; i < rowsPad; i++ {
		b.sq[i] = 0
		if i < n {
			if s, _ := quantRow(qa[i*D:(i+1)*D], q[i*ld:i*ld+D], b.zeros, true); s != 0 {
				b.sq[i] = s * scale
				continue
			}
		}
		fill128(b.qa[i*D : (i+1)*D])
	}

	// K, centered, its quadruplets of depths copied as words into the
	// panels of 16 keys.
	b.mean = grow(b.mean, D)
	clear(b.mean)
	for j := 0; j < n; j++ {
		AddTo(b.mean, k[j*ld:j*ld+D])
	}
	Scale(b.mean, 1/float32(n))
	b.kp = grow(b.kp, nPad/nr8*kq*nr8*4)
	clear(b.kp)
	b.ks = grow(b.ks, nPad)
	b.kzc = grow(b.kzc, nPad)
	clear(b.ks)
	clear(b.kzc)
	b.kb = grow(b.kb, 4*D)
	kb := asInt8(b.kb)
	kw := asUint32(b.kb[:D])
	pw := asUint32(asUint8(b.kp))
	for j := 0; j < n; j++ {
		s, sum := quantRow(kb[:D], k[j*ld:j*ld+D], b.mean, false)
		if s == 0 {
			continue
		}
		b.ks[j] = s
		b.kzc[j] = 128 * float32(sum)
		dst := pw[(j/nr8)*kq*nr8+j%nr8:]
		for d4, w := range kw {
			dst[d4*nr8] = w
		}
	}

	// V, quantized per dimension four keys at a time, interleaved into
	// the panels.
	b.vp = grow(b.vp, dPanels*kqn*nr8*4)
	clear(b.vp)
	b.vs = grow(b.vs, dPanels*nr8)
	clear(b.vs)
	for j := 0; j < n; j++ {
		MaxAbsTo(b.vs[:D], v[j*ld:j*ld+D])
	}
	b.vinv = grow(b.vinv, D)
	for d := 0; d < D; d++ {
		b.vinv[d] = 0
		if b.vs[d] > 0 {
			b.vs[d] /= 127
			b.vinv[d] = 1 / b.vs[d]
		}
	}
	for j0 := 0; j0 < n; j0 += 4 {
		for t := 0; t < 4; t++ {
			if j0+t < n {
				quantCols(kb[t*D:(t+1)*D], v[(j0+t)*ld:(j0+t)*ld+D], b.vinv)
			} else {
				clear(kb[t*D : (t+1)*D])
			}
		}
		interleave4(b.vp[(j0/4)*nr8*4:], kb, D, kqn*nr8*4)
	}
}

func asInt8(b []uint8) []int8 {
	return unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(b))), len(b))
}

func asUint8(b []int8) []uint8 {
	return unsafe.Slice((*uint8)(unsafe.Pointer(unsafe.SliceData(b))), len(b))
}

// asUint32 views b (a multiple of 4 bytes) as words.
func asUint32(b []uint8) []uint32 {
	return unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
}

// Attend writes into o the attention of the queries [i0, i1), row i at
// o[(i−i0)·ldo:], D values. i0 is a multiple of AttnBlock. window < 0:
// global attention; otherwise |i − j| ≤ window.
func (b *Attn8) Attend(o []float32, ldo, i0, i1, window int, sc *AttnScratch) {
	n, D, kq, lda, nPad, dPanels := b.n, b.D, b.kq, b.lda, b.nPad, b.dPanels
	kqn := nPad / 4
	if i0 >= i1 {
		return
	}
	sc.s = grow(sc.s, AttnBlock*nPad)
	sc.pa = grow(sc.pa, AttnBlock*nPad)
	sc.sp = grow(sc.sp, AttnBlock)
	base := i0
	var tile [mr8 * nr8]int32
	for end := i1; i0 < end; i0 += AttnBlock {
		i1 := min(end, i0+AttnBlock)
		// Keys the block sees, in whole panels: [c0, c1).
		c0, c1 := 0, nPad
		if window >= 0 {
			c0 = max(0, i0-window) / nr8 * nr8
			c1 = min(nPad, (i1-1+window+1+nr8-1)/nr8*nr8)
		}
		W := c1 - c0
		s := sc.s[:AttnBlock*W]
		// Scores: S = Q·(K − k̄)ᵀ. The padding rows of the last panel
		// are zero (quantizeA), their scores too.
		for r0 := i0; r0 < i1; r0 += mr8 {
			ap := b.qa[r0*lda : (r0+mr8)*lda]
			for p := c0 / nr8; p < c1/nr8; p++ {
				bp := b.kp[p*kq*nr8*4 : (p+1)*kq*nr8*4]
				out := s[(r0-i0)*W+p*nr8-c0:]
				if microKernel8Deq(kq, ap, lda, bp, out, W, b.ks[p*nr8:], b.kzc[p*nr8:], b.sq[r0:], false) {
					continue
				}
				microKernel8(kq, ap, lda, bp, &tile)
				for r := 0; r < mr8; r++ {
					Dequantize(out[r*W:r*W+nr8], tile[r*nr8:(r+1)*nr8], b.ks[p*nr8:(p+1)*nr8], b.kzc[p*nr8:(p+1)*nr8], b.sq[r0+r], false)
				}
			}
		}
		// Softmax on the keys of each row, then P quantized to uint8.
		pa := sc.pa[:AttnBlock*W]
		for r := 0; r < AttnBlock; r++ {
			i := i0 + r
			prow := pa[r*W : (r+1)*W]
			if i >= i1 {
				clear(prow) // padding rows of the last panel
				continue
			}
			row := s[r*W : (r+1)*W]
			cs, ce := 0, min(n, c1)-c0
			if window >= 0 {
				cs, ce = max(0, i-window-c0), min(n, i+window+1)-c0
			}
			valid := row[cs:ce]
			ExpShift(valid, MaxOf(valid))
			clear(prow[:cs])
			// Normalized by the sum of the quantized weights, those rounded
			// to zero left out: the weights used sum to exactly one.
			sc.sp[r] = 1 / float32(QuantizeRowU8(prow[cs:ce], valid, 255))
			clear(prow[ce:])
		}
		// O = P·V, normalized by the sum of each row.
		for r0 := i0; r0 < i1; r0 += mr8 {
			ap := pa[(r0-i0)*W : (r0-i0+mr8)*W]
			rows := min(mr8, i1-r0)
			for p := 0; p < dPanels; p++ {
				bp := b.vp[p*kqn*nr8*4+(c0/4)*nr8*4 : p*kqn*nr8*4+(c1/4)*nr8*4]
				cols := min(nr8, D-p*nr8)
				if rows == mr8 && cols == nr8 &&
					microKernel8Deq(W/4, ap, W, bp, o[(r0-base)*ldo+p*nr8:], ldo, b.vs[p*nr8:], b.zeros, sc.sp[r0-i0:], false) {
					continue
				}
				microKernel8(W/4, ap, W, bp, &tile)
				for r := 0; r < rows; r++ {
					i := r0 + r
					Dequantize(o[(i-base)*ldo+p*nr8:(i-base)*ldo+p*nr8+cols], tile[r*nr8:r*nr8+cols], b.vs[p*nr8:p*nr8+cols], b.zeros[:cols], sc.sp[i-i0], false)
				}
			}
		}
	}
}
