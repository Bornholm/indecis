package linerec

import (
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// State holds a line's activations: what the backward pass reads, and the
// logits. A State is reused from line to line; its slices grow as needed.
type State struct {
	W  int // width of the line in pixels
	L  int // valid steps: ⌈W/Stride⌉
	Tc int // steps the context branch computes: one past the line when the padding has it

	StemIn []float32 // [L, Height·Stride] patches of the middle band
	CtxIn  []float32 // [R·Tc, Stride·Stride] patches of the window, row-major over (r, t)
	CtxA   []float32 // [R·Tc, CtxChannels] patch projection, before HardSwish
	CtxH   []float32 // its HardSwish
	CtxD   []float32 // [R·Tc, CtxChannels] after the 3×3 depthwise convolution
	CtxB   []float32 // [R·Tc, CtxDim] pointwise convolution, before HardSwish
	CtxF   []float32 // its HardSwish
	CtxP   []float32 // [R·Tc, CtxHeads] attention scores
	Att    []float32 // [L, CtxHeads, R] attention over the rows: softmax of the scores plus the row prior
	Pooled []float32 // [L, CtxDim]
	Pre    []float32 // [L, Dim] stem + context, before normalization
	Normed []float32 // [L, Dim] after normalization, before HardSwish
	Mean   []float32 // [L] layer norm: mean of each step
	Rstd   []float32 // [L] layer norm: 1/σ of each step

	Blocks []BlockState
	H      []float32 // [L, Dim] output of the last block
	Logits []float32 // [L, Vocab]

	work work // gradients of the activations, for Backward
}

// BlockState holds a block's activations.
type BlockState struct {
	In   []float32 // [L, Dim]
	D    []float32 // [L, Dim] depthwise convolution
	E    []float32 // [L, Expand] expansion, before HardSwish
	U    []float32 // its HardSwish
	Mean []float32 // [Expand] mean of U over the steps
	Z    []float32 // [SE] fc1, before ReLU
	A    []float32 // [SE] its ReLU
	Gate []float32 // [Expand] sigmoid of fc2
	V    []float32 // [L, Expand] U·Gate
}

// Forward computes the logits of a line. ink is the line's ink map,
// [Rows, w] row-major (0 = paper, 1 = ink); the pixels past w count as
// paper. The logits of the L valid steps are in s.Logits.
func (m *Model) Forward(s *State, ink []float32, w int) {
	c := m.Cfg
	if w <= 0 || len(ink) < c.Rows()*w {
		panic("linerec: Forward: ink map too short")
	}
	T, L := c.Steps(w)
	s.W, s.L, s.Tc = w, L, min(T, L+1)

	m.stem(s, ink)
	m.context(s, ink)
	m.normalize(s)

	if cap(s.Blocks) < c.Blocks {
		s.Blocks = make([]BlockState, c.Blocks)
	}
	s.Blocks = s.Blocks[:c.Blocks]
	s.H = grow(s.H, L*c.Dim)
	in := s.H
	if c.Blocks > 0 {
		s.Blocks[0].In = grow(s.Blocks[0].In, L*c.Dim)
		in = s.Blocks[0].In
	}
	linalg.HardSwish(in, s.Normed)
	for i := range m.Blocks {
		out := s.H
		if i+1 < len(m.Blocks) {
			s.Blocks[i+1].In = grow(s.Blocks[i+1].In, L*c.Dim)
			out = s.Blocks[i+1].In
		}
		m.block(&m.Blocks[i], &s.Blocks[i], out, L)
	}

	s.Logits = grow(s.Logits, L*c.Vocab)
	dense(s.Logits, s.H, m.Head, m.HeadBias, L, c.Dim, c.Vocab)
}

// stem projects the middle band of each valid step.
func (m *Model) stem(s *State, ink []float32) {
	c := m.Cfg
	S, Hh, L, w := c.Stride, c.Height, s.L, s.W
	top := (c.Window / 2) * c.Height
	k := Hh * S
	s.StemIn = grow(s.StemIn, L*k)
	for t := range L {
		p := s.StemIn[t*k : (t+1)*k]
		for y := range Hh {
			row := ink[(top+y)*w : (top+y+1)*w]
			for x := range S {
				p[y*S+x] = pixel(row, t*S+x)
			}
		}
	}
	s.Pre = grow(s.Pre, L*c.Dim)
	dense(s.Pre, s.StemIn, m.Stem, m.StemBias, L, k, c.Dim)
}

func pixel(row []float32, x int) float32 {
	if x < len(row) {
		return row[x]
	}
	return 0
}

// context pools the whole window over its rows and adds the result to
// s.Pre.
func (m *Model) context(s *State, ink []float32) {
	c := m.Cfg
	S, R, Tc, L, w := c.Stride, c.CtxRows(), s.Tc, s.L, s.W
	CC, CD, NH := c.CtxChannels, c.CtxDim, c.CtxHeads
	n := R * Tc
	k := S * S

	s.CtxIn = grow(s.CtxIn, n*k)
	for r := range R {
		for t := range Tc {
			p := s.CtxIn[(r*Tc+t)*k : (r*Tc+t+1)*k]
			for y := range S {
				row := ink[(r*S+y)*w : (r*S+y+1)*w]
				for x := range S {
					p[y*S+x] = pixel(row, t*S+x)
				}
			}
		}
	}
	s.CtxA = grow(s.CtxA, n*CC)
	s.CtxH = grow(s.CtxH, n*CC)
	denseHS(s.CtxA, s.CtxH, s.CtxIn, m.CtxPatch, m.CtxPatchBias, n, k, CC)

	s.CtxD = grow(s.CtxD, n*CC)
	depthwise3x3(s.CtxD, s.CtxH, m.CtxDW.kernelT(9, CC), R, Tc, CC)

	s.CtxB = grow(s.CtxB, n*CD)
	s.CtxF = grow(s.CtxF, n*CD)
	denseHS(s.CtxB, s.CtxF, s.CtxD, m.CtxPW, m.CtxPWBias, n, CC, CD)

	s.CtxP = grow(s.CtxP, n*NH)
	dense(s.CtxP, s.CtxF, m.CtxScore, m.CtxScoreBias, n, CD, NH)
	prior := m.RowPrior.W // [NH, R]
	hd := CD / NH
	s.Att = grow(s.Att, L*NH*R)
	s.Pooled = grow(s.Pooled, L*CD)
	clear(s.Pooled)
	for t := range L {
		for h := range NH {
			a := s.Att[(t*NH+h)*R : (t*NH+h+1)*R]
			for r := range R {
				a[r] = s.CtxP[(r*Tc+t)*NH+h] + prior[h*R+r]
			}
			linalg.Scale(a, 1/linalg.ExpShift(a, linalg.MaxOf(a)))
			o := s.Pooled[t*CD+h*hd : t*CD+(h+1)*hd]
			for r, ar := range a {
				f := s.CtxF[(r*Tc+t)*CD+h*hd : (r*Tc+t)*CD+(h+1)*hd]
				for j := range o {
					o[j] += ar * f[j]
				}
			}
		}
	}
	// The pooled steps past L are not needed: only the valid steps continue.
	matMul(s.Pre, s.Pooled, m.CtxProj, L, CD, c.Dim, true)
	addBias(s.Pre, m.CtxProjBias.W, L, c.Dim)
}

// normalize applies the stem normalization to s.Pre into s.Normed.
func (m *Model) normalize(s *State) {
	c := m.Cfg
	D, L := c.Dim, s.L
	s.Normed = grow(s.Normed, L*D)
	g, b := m.NormW.W, m.NormB.W
	if m.Cfg.Norm == "batch" {
		for t := range L {
			x, y := s.Pre[t*D:(t+1)*D], s.Normed[t*D:(t+1)*D]
			for j := range D {
				inv := float32(1 / math.Sqrt(float64(m.RunVar.W[j])+normEps))
				y[j] = (x[j]-m.RunMean.W[j])*inv*g[j] + b[j]
			}
		}
		return
	}
	s.Mean = grow(s.Mean, L)
	s.Rstd = grow(s.Rstd, L)
	for t := range L {
		x, y := s.Pre[t*D:(t+1)*D], s.Normed[t*D:(t+1)*D]
		var mean float64
		for _, v := range x {
			mean += float64(v)
		}
		mean /= float64(D)
		var v2 float64
		for _, v := range x {
			d := float64(v) - mean
			v2 += d * d
		}
		rstd := 1 / math.Sqrt(v2/float64(D)+normEps)
		s.Mean[t], s.Rstd[t] = float32(mean), float32(rstd)
		for j := range D {
			y[j] = float32((float64(x[j])-mean)*rstd)*g[j] + b[j]
		}
	}
}

// block runs a depthwise block from b.In into out.
func (m *Model) block(p *Block, b *BlockState, out []float32, L int) {
	c := m.Cfg
	D, E := c.Dim, c.Expand
	b.D = grow(b.D, L*D)
	depthwise1D(b.D, b.In, p.DW.kernelT(c.Kernel, D), p.DWBias.W, L, D, c.Kernel)

	b.E = grow(b.E, L*E)
	b.U = grow(b.U, L*E)
	denseHS(b.E, b.U, b.D, p.Expand, p.ExpandBias, L, D, E)

	// Squeeze-excitation over the valid steps.
	b.Mean = grow(b.Mean, E)
	clear(b.Mean)
	for t := range L {
		linalg.AddTo(b.Mean, b.U[t*E:(t+1)*E])
	}
	linalg.Scale(b.Mean, 1/float32(L))
	b.Z = grow(b.Z, c.SE)
	dense(b.Z, b.Mean, p.FC1, p.FC1Bias, 1, E, c.SE)
	b.A = grow(b.A, c.SE)
	for i, v := range b.Z {
		b.A[i] = max(v, 0)
	}
	b.Gate = grow(b.Gate, E)
	dense(b.Gate, b.A, p.FC2, p.FC2Bias, 1, c.SE, E)
	sigmoid(b.Gate)
	b.V = grow(b.V, L*E)
	for t := range L {
		linalg.MulTo(b.V[t*E:(t+1)*E], b.U[t*E:(t+1)*E], b.Gate)
	}

	// Residual: out = In + V·Projectᵀ + bias.
	copy(out[:L*D], b.In[:L*D])
	matMul(out, b.V, p.Project, L, E, D, true)
	addBias(out, p.ProjBias.W, L, D)
}

// dense computes y = x·Wᵀ + bias: x [n, k], W [out, k] (a pointwise
// convolution's kernel), y [n, out]. The bias is written first and the
// product accumulated onto it: linalg's accumulating store is faster than
// its plain one for these narrow products.
func dense(y, x []float32, w, bias *Param, n, k, out int) {
	for i := range n {
		copy(y[i*out:(i+1)*out], bias.W)
	}
	matMul(y, x, w, n, k, out, true)
}

// denseHS is dense followed by HardSwish: y = x·Wᵀ + bias, a = HardSwish(y).
func denseHS(y, a, x []float32, w, bias *Param, n, k, out int) {
	dense(y, x, w, bias, n, k, out)
	linalg.HardSwish(a[:n*out], y[:n*out])
}

// matMul computes y (+)= x·Wᵀ, with W packed when the model is prepared.
func matMul(y, x []float32, w *Param, n, k, out int, accumulate bool) {
	if w.packed != nil {
		linalg.MatMulPacked(y, x, w.packed, n, accumulate)
		return
	}
	linalg.MatMul(y, x, w.W, n, k, out, false, true, accumulate)
}

func addBias(y, bias []float32, n, out int) {
	for i := range n {
		linalg.AddTo(y[i*out:(i+1)*out], bias)
	}
}

// depthwise1D convolves each channel over the steps: y[t][c] = bias[c] +
// Σ_k w[k][c]·x[t+k-K/2][c], zero outside [0, L). w is the kernel
// transposed, [K, C].
func depthwise1D(y, x, w, bias []float32, L, C, K int) {
	half := K / 2
	for t := range L {
		o := y[t*C : (t+1)*C]
		copy(o, bias)
		for k := range K {
			tt := t + k - half
			if tt < 0 || tt >= L {
				continue
			}
			linalg.MulAddTo(o, w[k*C:(k+1)*C], x[tt*C:(tt+1)*C])
		}
	}
}

// depthwise3x3 convolves each channel of x [R, T, C] with its 3×3 kernel
// (w transposed, [9, C]), zero padding, no bias.
func depthwise3x3(y, x, w []float32, R, T, C int) {
	for r := range R {
		for t := range T {
			o := y[(r*T+t)*C : (r*T+t+1)*C]
			clear(o)
			for dr := range 3 {
				rr := r + dr - 1
				if rr < 0 || rr >= R {
					continue
				}
				for dt := range 3 {
					tt := t + dt - 1
					if tt < 0 || tt >= T {
						continue
					}
					k := dr*3 + dt
					linalg.MulAddTo(o, w[k*C:(k+1)*C], x[(rr*T+tt)*C:(rr*T+tt+1)*C])
				}
			}
		}
	}
}

func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

// sigmoid replaces x by 1/(1+e^-x), with linalg's SIMD exponential: Go's
// scalar math.Exp after AVX code is several times slower.
func sigmoid(x []float32) {
	for i, v := range x {
		x[i] = -v
	}
	linalg.ExpShift(x, 0)
	for i, e := range x {
		x[i] = 1 / (1 + e)
	}
}
