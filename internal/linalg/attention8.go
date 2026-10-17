package linalg

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

// attnRows is the number of queries scored at a time: a multiple of mr8.
const attnRows = 8 * mr8

// AttnBuffers holds the buffers of Attend8, reused from call to call.
type AttnBuffers struct {
	qa      []uint8   // quantized Q, rows of lda bytes
	sq      []float32 // scale of each row of Q (times the score scale)
	kp      []int8    // packed centered K: panels of 16 keys, depth D
	ks, kzc []float32 // scale and 128·sum of each key
	vp      []int8    // packed V: panels of 16 dimensions, depth the keys
	vs      []float32 // scale of each dimension of V
	zeros   []float32
	mean    []float32
	kc      []float32 // a centered key
	s       []float32 // scores of a block of queries
	pa      []uint8   // quantized P of a block of queries
	sp      []float32 // scale of each row of P (1/sum of its quantized weights)
	ints    []int32
	off     []int // offset of each depth in a packed panel
	vinv    []float32
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
	if n == 0 {
		return
	}
	kq := (D + 3) / 4
	lda := kq * 4
	rowsPad := (n + mr8 - 1) / mr8 * mr8
	keyPanels := (n + nr8 - 1) / nr8
	nPad := keyPanels * nr8 // keys, padded to whole panels
	kqn := nPad / 4         // depth of P·V in quadruplets

	// Q, quantized per row; the score scale goes into the row scales.
	b.qa = grow(b.qa, rowsPad*lda)
	b.sq = grow(b.sq, rowsPad)
	quantizeA(b.qa, b.sq, q, n, D, lda)
	for i := range b.sq {
		b.sq[i] *= scale
	}

	// K, centered then quantized per key.
	b.mean = grow(b.mean, D)
	clear(b.mean)
	for j := 0; j < n; j++ {
		AddTo(b.mean, k[j*D:(j+1)*D])
	}
	Scale(b.mean, 1/float32(n))
	b.kp = grow(b.kp, keyPanels*kq*nr8*4)
	clear(b.kp)
	b.ks = grow(b.ks, nPad)
	b.kzc = grow(b.kzc, nPad)
	clear(b.ks)
	clear(b.kzc)
	b.kc = grow(b.kc, D)
	b.off = grow(b.off, max(D, nr8))
	for d := 0; d < D; d++ {
		b.off[d] = (d/4)*nr8*4 + d%4
	}
	for j := 0; j < n; j++ {
		kc := b.kc
		for d, x := range k[j*D : (j+1)*D] {
			kc[d] = x - b.mean[d]
		}
		mx := AbsMax(kc)
		if mx == 0 {
			continue
		}
		s := mx / 127
		inv := 1 / s
		b.ks[j] = s
		dst := b.kp[(j/nr8)*kq*nr8*4+(j%nr8)*4:]
		var sum int32
		for d, x := range kc {
			qv := int8(roundHalfEven(x * inv))
			dst[b.off[d]] = qv
			sum += int32(qv)
		}
		b.kzc[j] = 128 * float32(sum)
	}

	// V, quantized per dimension, its depth (the keys) padded with zeros.
	dPanels := (D + nr8 - 1) / nr8
	Dp := dPanels * nr8
	b.vp = grow(b.vp, dPanels*kqn*nr8*4)
	clear(b.vp)
	b.vs = grow(b.vs, Dp)
	clear(b.vs)
	for j := 0; j < n; j++ {
		for d, x := range v[j*D : (j+1)*D] {
			b.vs[d] = max(b.vs[d], abs32(x))
		}
	}
	b.vinv = grow(b.vinv, D)
	for d := 0; d < D; d++ {
		b.vinv[d] = 0
		if b.vs[d] > 0 {
			b.vs[d] /= 127
			b.vinv[d] = 1 / b.vs[d]
		}
		b.off[d] = (d/nr8)*kqn*nr8*4 + (d%nr8)*4
	}
	for j := 0; j < n; j++ {
		dst := b.vp[(j/4)*nr8*4+j%4:]
		for d, x := range v[j*D : (j+1)*D] {
			dst[b.off[d]] = int8(roundHalfEven(x * b.vinv[d]))
		}
	}
	b.zeros = grow(b.zeros, Dp)
	clear(b.zeros)

	b.s = grow(b.s, attnRows*nPad)
	b.pa = grow(b.pa, attnRows*nPad)
	b.sp = grow(b.sp, attnRows)
	var tile [mr8 * nr8]int32
	for i0 := 0; i0 < n; i0 += attnRows {
		i1 := min(n, i0+attnRows)
		// Keys the block sees, in whole panels: [c0, c1).
		c0, c1 := 0, nPad
		if window >= 0 {
			c0 = max(0, i0-window) / nr8 * nr8
			c1 = min(nPad, (i1-1+window+1+nr8-1)/nr8*nr8)
		}
		W := c1 - c0
		s := b.s[:attnRows*W]
		// Scores: S = Q·(K − k̄)ᵀ, a panel of rows at a time, rescaled
		// row by row in one call.
		acc := grow(b.ints, mr8*max(W, Dp))
		b.ints = acc
		for r0 := i0; r0 < i1; r0 += mr8 {
			ap := b.qa[r0*lda : (r0+mr8)*lda]
			for p := c0 / nr8; p < c1/nr8; p++ {
				microKernel8(kq, ap, lda, b.kp[p*kq*nr8*4:(p+1)*kq*nr8*4], &tile)
				jc := p*nr8 - c0
				for r := 0; r < mr8; r++ {
					copy(acc[r*W+jc:r*W+jc+nr8], tile[r*nr8:(r+1)*nr8])
				}
			}
			for r := 0; r < min(mr8, i1-r0); r++ {
				i := r0 + r
				Dequantize(s[(i-i0)*W:(i-i0+1)*W], acc[r*W:(r+1)*W], b.ks[c0:c1], b.kzc[c0:c1], b.sq[i], false)
			}
		}
		// Softmax on the keys of each row, then P quantized to uint8.
		pa := b.pa[:attnRows*W]
		for r := 0; r < attnRows; r++ {
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
			b.sp[r] = 1 / float32(QuantizeRowU8(prow[cs:ce], valid, 255))
			clear(prow[ce:])
		}
		// O = P·V, normalized by the sum of each row.
		for r0 := i0; r0 < i1; r0 += mr8 {
			ap := pa[(r0-i0)*W : (r0-i0+mr8)*W]
			for p := 0; p < dPanels; p++ {
				bp := b.vp[p*kqn*nr8*4+(c0/4)*nr8*4 : p*kqn*nr8*4+(c1/4)*nr8*4]
				microKernel8(W/4, ap, W, bp, &tile)
				for r := 0; r < mr8; r++ {
					copy(acc[r*Dp+p*nr8:r*Dp+(p+1)*nr8], tile[r*nr8:(r+1)*nr8])
				}
			}
			for r := 0; r < min(mr8, i1-r0); r++ {
				i := r0 + r
				Dequantize(o[i*D:(i+1)*D], acc[r*Dp:r*Dp+D], b.vs[:D], b.zeros[:D], b.sp[i-i0], false)
			}
		}
	}
}
