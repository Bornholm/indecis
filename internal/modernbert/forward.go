package modernbert

import (
	"fmt"
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// Batch is a batch of sequences right-padded to T tokens.
type Batch struct {
	IDs  []int32 // [B·T]
	Lens []int   // actual length of each sequence
	T    int
}

// NewBatch pads the sequences with pad.
func NewBatch(seqs [][]int32, pad int32) Batch {
	T := 0
	for _, s := range seqs {
		T = max(T, len(s))
	}
	b := Batch{IDs: make([]int32, len(seqs)*T), Lens: make([]int, len(seqs)), T: T}
	for i, s := range seqs {
		copy(b.IDs[i*T:], s)
		for j := len(s); j < T; j++ {
			b.IDs[i*T+j] = pad
		}
		b.Lens[i] = len(s)
	}
	return b
}

// B is the number of sequences.
func (b Batch) B() int { return len(b.Lens) }

// State holds the activations of a forward pass, read by the backward pass.
type State struct {
	Batch  Batch
	N      int       // B·T rows
	emb    []float32 // raw embeddings, [N, H]
	embLN  lnCache
	layers []layerState
	final  lnCache
	// Hidden is the encoder output after final_norm, [N, H].
	Hidden []float32
}

type layerState struct {
	in     []float32 // residual input, [N, H]
	attnIn []float32 // attn_norm(in), or in for layer 0
	attnLN lnCache
	q, k   []float32 // after RoPE, [B, heads, T, D]
	v      []float32 // [B, heads, T, D]
	p      []float32 // attention probabilities, block (b, h) of len×len
	ctx    []float32 // concatenated heads, input to Wo, [N, H]
	h1     []float32 // in + attention
	mlpIn  []float32 // mlp_norm(h1)
	mlpLN  lnCache
	z      []float32 // output of Wi, [N, 2·I]
	g      []float32 // gelu(input) ⊙ gate, input to WoMLP, [N, I]
	out    []float32 // h1 + MLP
}

// Forward encodes a batch. The returned State is used by the backward
// pass; in inference, only State.Hidden is needed.
func (m *Model) Forward(b Batch) (*State, error) {
	cfg := m.Cfg
	H := cfg.Hidden
	N := b.B() * b.T
	if len(b.IDs) != N {
		return nil, fmt.Errorf("modernbert: inconsistent batch")
	}
	for _, id := range b.IDs {
		if id < 0 || int(id) >= cfg.Vocab {
			return nil, fmt.Errorf("modernbert: id %d out of vocabulary", id)
		}
	}
	m.restoreWeights()
	s := &State{Batch: b, N: N}

	s.emb = make([]float32, N*H)
	for r, id := range b.IDs {
		m.embRow(id, s.emb[r*H:(r+1)*H])
	}
	x := make([]float32, N*H)
	layerNorm(x, s.emb, m.EmbNorm.W, N, H, cfg.NormEps, &s.embLN)

	s.layers = make([]layerState, cfg.Layers)
	for l := range m.Layers {
		ls := &s.layers[l]
		m.layerForward(l, b, x, ls)
		x = ls.out
	}

	s.Hidden = make([]float32, N*H)
	layerNorm(s.Hidden, x, m.FinalNorm.W, N, H, cfg.NormEps, &s.final)
	return s, nil
}

func (m *Model) layerForward(l int, b Batch, x []float32, ls *layerState) {
	cfg := m.Cfg
	L := m.Layers[l]
	H, I := cfg.Hidden, cfg.Intermediate
	N := b.B() * b.T

	ls.in = x
	if L.AttnNorm == nil {
		ls.attnIn = x
	} else {
		ls.attnIn = make([]float32, N*H)
		layerNorm(ls.attnIn, x, L.AttnNorm.W, N, H, cfg.NormEps, &ls.attnLN)
	}

	qkv := make([]float32, N*3*H)
	linalg.MatMul(qkv, ls.attnIn, L.Wqkv.W, N, H, 3*H, false, true, false)
	m.attentionForward(l, b, qkv, ls)

	attnOut := make([]float32, N*H)
	linalg.MatMul(attnOut, ls.ctx, L.Wo.W, N, H, H, false, true, false)
	ls.h1 = make([]float32, N*H)
	for i := range ls.h1 {
		ls.h1[i] = x[i] + attnOut[i]
	}

	ls.mlpIn = make([]float32, N*H)
	layerNorm(ls.mlpIn, ls.h1, L.MLPNorm.W, N, H, cfg.NormEps, &ls.mlpLN)
	ls.z = make([]float32, N*2*I)
	linalg.MatMul(ls.z, ls.mlpIn, L.Wi.W, N, H, 2*I, false, true, false)
	ls.g = make([]float32, N*I)
	gluForward(ls.g, ls.z, N, I)
	mlpOut := make([]float32, N*H)
	linalg.MatMul(mlpOut, ls.g, L.WoMLP.W, N, I, H, false, true, false)

	ls.out = make([]float32, N*H)
	for i := range ls.out {
		ls.out[i] = ls.h1[i] + mlpOut[i]
	}
}

// theta returns the RoPE base of layer l.
func (m *Model) theta(l int) float64 {
	if m.Cfg.IsGlobal(l) {
		return m.Cfg.GlobalTheta
	}
	return m.Cfg.LocalTheta
}

// attentionForward splits qkv by head, applies RoPE, then computes the
// attention of each (sequence, head) over its actual length.
func (m *Model) attentionForward(l int, b Batch, qkv []float32, ls *layerState) {
	cfg := m.Cfg
	H, nh, D := cfg.Hidden, cfg.Heads, cfg.HeadDim()
	B, T := b.B(), b.T
	half := D / 2
	rope := m.ropeFor(m.theta(l), T)

	ls.q = make([]float32, B*nh*T*D)
	ls.k = make([]float32, B*nh*T*D)
	ls.v = make([]float32, B*nh*T*D)
	linalg.Parallel(B*T, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			bi, t := r/T, r%T
			row := qkv[r*3*H : (r+1)*3*H]
			cos := rope.cos[t*half : (t+1)*half]
			sin := rope.sin[t*half : (t+1)*half]
			for h := 0; h < nh; h++ {
				dst := ((bi*nh+h)*T + t) * D
				q := ls.q[dst : dst+D]
				k := ls.k[dst : dst+D]
				copy(q, row[h*D:(h+1)*D])
				copy(k, row[H+h*D:H+(h+1)*D])
				copy(ls.v[dst:dst+D], row[2*H+h*D:2*H+(h+1)*D])
				applyRope(q, cos, sin)
				applyRope(k, cos, sin)
			}
		}
	})

	ls.p = make([]float32, B*nh*T*T)
	ls.ctx = make([]float32, B*T*H)
	scale := float32(1 / math.Sqrt(float64(D)))
	window := -1
	if !cfg.IsGlobal(l) {
		window = cfg.Window()
	}
	linalg.Parallel(B*nh, 1, func(lo, hi int) {
		o := make([]float32, T*D)
		for task := lo; task < hi; task++ {
			bi, h := task/nh, task%nh
			n := b.Lens[bi]
			if n == 0 {
				continue
			}
			base := (bi*nh + h) * T * D
			q := ls.q[base : base+n*D]
			k := ls.k[base : base+n*D]
			v := ls.v[base : base+n*D]
			p := ls.p[(bi*nh+h)*T*T : (bi*nh+h)*T*T+n*n]
			linalg.MatMulSerial(p, q, k, n, D, n, false, true, false)
			negInf := float32(math.Inf(-1))
			for i := 0; i < n; i++ {
				row := p[i*n : (i+1)*n]
				for j := range row {
					if window >= 0 && (i-j > window || j-i > window) {
						row[j] = negInf
					} else {
						row[j] *= scale
					}
				}
			}
			softmaxRows(p, n, n)
			linalg.MatMulSerial(o, p, v, n, n, D, false, false, false)
			for t := 0; t < n; t++ {
				copy(ls.ctx[(bi*T+t)*H+h*D:(bi*T+t)*H+(h+1)*D], o[t*D:(t+1)*D])
			}
		}
	})
}

// MeanPool averages the hidden states of each sequence over its actual
// tokens, special tokens included, like sentence-transformers pooling.
func MeanPool(hidden []float32, b Batch, h int) []float32 {
	out := make([]float32, b.B()*h)
	for bi, n := range b.Lens {
		if n == 0 {
			continue
		}
		acc := make([]float64, h)
		for t := 0; t < n; t++ {
			row := hidden[(bi*b.T+t)*h : (bi*b.T+t+1)*h]
			for i, v := range row {
				acc[i] += float64(v)
			}
		}
		for i := range acc {
			out[bi*h+i] = float32(acc[i] / float64(n))
		}
	}
	return out
}
