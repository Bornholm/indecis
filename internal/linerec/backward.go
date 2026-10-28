package linerec

import (
	"math"

	"github.com/bornholm/indecis/linalg"
	"github.com/bornholm/indecis/optim"
)

// Grads holds a gradient buffer for each trainable parameter, in the order
// of Model.Params. Lines can be backpropagated in parallel, each worker
// into its own Grads, summed afterwards.
type Grads struct {
	G   [][]float32
	idx map[*Param]int
}

// NewGrads allocates zero gradients for m's parameters.
func (m *Model) NewGrads() *Grads {
	ps := m.Params()
	g := &Grads{G: make([][]float32, len(ps)), idx: make(map[*Param]int, len(ps))}
	for i, p := range ps {
		g.G[i] = make([]float32, len(p.W))
		g.idx[p] = i
	}
	return g
}

func (g *Grads) of(p *Param) []float32 { return g.G[g.idx[p]] }

// Zero resets the gradients.
func (g *Grads) Zero() {
	for _, b := range g.G {
		clear(b)
	}
}

// Add adds o's gradients to g.
func (g *Grads) Add(o *Grads) {
	for i, b := range g.G {
		linalg.AddTo(b, o.G[i])
	}
}

// work holds the gradients of the activations during a backward pass.
type work struct {
	dH, dIn, dD, dE, dV []float32
	dMean, dZ, dA, dG   []float32
	dPre, dNorm         []float32
	dPooled, dAtt       []float32
	dF, dB, dCD, dCH    []float32
	dS                  []float32
}

// Backward backpropagates dLogits ([L, Vocab], the gradient of the loss
// with respect to s.Logits) through the line computed by the last Forward
// on s, and accumulates the parameter gradients into g.
func (m *Model) Backward(s *State, dLogits []float32, g *Grads) {
	c := m.Cfg
	L, D := s.L, c.Dim
	w := &s.work

	// Head.
	w.dH = grow(w.dH, L*D)
	linalg.MatMul(w.dH, dLogits, m.Head.W, L, c.Vocab, D, false, false, false)
	weightGrad(g.of(m.Head), g.of(m.HeadBias), dLogits, s.H, L, D, c.Vocab)

	// Blocks, last to first: w.dH is the gradient of the block's output,
	// then of its input.
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		m.blockBackward(&m.Blocks[i], &s.Blocks[i], w, g, L)
	}

	// HardSwish after the normalization.
	w.dNorm = grow(w.dNorm, L*D)
	hardSwishGrad(w.dNorm, w.dH, s.Normed)

	// Normalization.
	w.dPre = grow(w.dPre, L*D)
	m.normBackward(s, w, g)

	// Stem: Pre = StemIn·Stemᵀ + b + Pooled·CtxProjᵀ + b.
	k := c.Height * c.Stride
	weightGrad(g.of(m.Stem), g.of(m.StemBias), w.dPre, s.StemIn, L, k, D)
	weightGrad(g.of(m.CtxProj), g.of(m.CtxProjBias), w.dPre, s.Pooled, L, c.CtxDim, D)
	w.dPooled = grow(w.dPooled, L*c.CtxDim)
	linalg.MatMul(w.dPooled, w.dPre, m.CtxProj.W, L, D, c.CtxDim, false, false, false)

	m.contextBackward(s, w, g)
}

// weightGrad accumulates the gradients of y = x·Wᵀ + bias: dW += dyᵀ·x,
// dbias += Σ dy. dy [n, out], x [n, k].
func weightGrad(dW, dBias, dy, x []float32, n, k, out int) {
	linalg.MatMul(dW, dy, x, out, n, k, true, false, true)
	for i := range n {
		linalg.AddTo(dBias, dy[i*out:(i+1)*out])
	}
}

// hardSwishGrad writes dx = dy·HardSwish'(x).
func hardSwishGrad(dx, dy, x []float32) {
	for i, v := range x[:len(dx)] {
		switch {
		case v <= -3:
			dx[i] = 0
		case v >= 3:
			dx[i] = dy[i]
		default:
			dx[i] = dy[i] * (v/3 + 0.5)
		}
	}
}

func (m *Model) blockBackward(p *Block, b *BlockState, w *work, g *Grads, L int) {
	c := m.Cfg
	D, E, SE := c.Dim, c.Expand, c.SE
	dOut := w.dH

	// out = In + V·Projectᵀ + bias.
	weightGrad(g.of(p.Project), g.of(p.ProjBias), dOut, b.V, L, E, D)
	w.dV = grow(w.dV, L*E)
	linalg.MatMul(w.dV, dOut, p.Project.W, L, D, E, false, false, false)

	// V = U·Gate: dU = dV·Gate, dGate = Σ_t dV·U.
	w.dG = grow(w.dG, E)
	clear(w.dG)
	w.dE = grow(w.dE, L*E) // dU for now
	for t := range L {
		dv, u := w.dV[t*E:(t+1)*E], b.U[t*E:(t+1)*E]
		linalg.MulTo(w.dE[t*E:(t+1)*E], dv, b.Gate)
		linalg.MulAddTo(w.dG, dv, u)
	}
	// Gate = σ(A·FC2ᵀ + b2).
	for i, gt := range b.Gate {
		w.dG[i] *= gt * (1 - gt)
	}
	weightGrad(g.of(p.FC2), g.of(p.FC2Bias), w.dG, b.A, 1, SE, E)
	w.dA = grow(w.dA, SE)
	linalg.MatMul(w.dA, w.dG, p.FC2.W, 1, E, SE, false, false, false)
	// A = ReLU(Z), Z = Mean·FC1ᵀ + b1.
	w.dZ = grow(w.dZ, SE)
	for i, z := range b.Z {
		if z > 0 {
			w.dZ[i] = w.dA[i]
		} else {
			w.dZ[i] = 0
		}
	}
	weightGrad(g.of(p.FC1), g.of(p.FC1Bias), w.dZ, b.Mean, 1, E, SE)
	w.dMean = grow(w.dMean, E)
	linalg.MatMul(w.dMean, w.dZ, p.FC1.W, 1, SE, E, false, false, false)
	// Mean = Σ_t U / L.
	linalg.Scale(w.dMean, 1/float32(L))
	for t := range L {
		linalg.AddTo(w.dE[t*E:(t+1)*E], w.dMean)
	}
	// U = HardSwish(E).
	hardSwishGrad(w.dE, w.dE, b.E)

	// E = D·Expandᵀ + b.
	weightGrad(g.of(p.Expand), g.of(p.ExpandBias), w.dE, b.D, L, D, E)
	w.dD = grow(w.dD, L*D)
	linalg.MatMul(w.dD, w.dE, p.Expand.W, L, E, D, false, false, false)

	// D = depthwise(In): the input's gradient adds to the residual's.
	w.dIn = grow(w.dIn, L*D)
	copy(w.dIn, dOut[:L*D])
	depthwise1DGrad(w.dIn, g.of(p.DW), g.of(p.DWBias), w.dD, b.In, p.DW.W, L, D, c.Kernel)
	w.dH, w.dIn = w.dIn, w.dH
}

// depthwise1DGrad accumulates the gradients of depthwise1D: dx, the
// kernel's dw [C, K] and dbias. w is the kernel in its stored layout
// [C, 1, K].
func depthwise1DGrad(dx, dw, dBias, dy, x, w []float32, L, C, K int) {
	half := K / 2
	for t := range L {
		d := dy[t*C : (t+1)*C]
		linalg.AddTo(dBias, d)
		for k := range K {
			tt := t + k - half
			if tt < 0 || tt >= L {
				continue
			}
			in := x[tt*C : (tt+1)*C]
			out := dx[tt*C : (tt+1)*C]
			for ch := range C {
				dw[ch*K+k] += d[ch] * in[ch]
				out[ch] += w[ch*K+k] * d[ch]
			}
		}
	}
}

func (m *Model) normBackward(s *State, w *work, g *Grads) {
	c := m.Cfg
	D, L := c.Dim, s.L
	gw, gb := g.of(m.NormW), g.of(m.NormB)
	gamma := m.NormW.W
	linalg.ZeroUpper() // scalar below
	if c.Norm == "batch" {
		for t := range L {
			dy, x, dx := w.dNorm[t*D:(t+1)*D], s.Pre[t*D:(t+1)*D], w.dPre[t*D:(t+1)*D]
			for j := range D {
				inv := float32(1 / math.Sqrt(float64(m.RunVar.W[j])+normEps))
				xhat := (x[j] - m.RunMean.W[j]) * inv
				gw[j] += dy[j] * xhat
				gb[j] += dy[j]
				dx[j] = dy[j] * gamma[j] * inv
			}
		}
		return
	}
	for t := range L {
		dy, x, dx := w.dNorm[t*D:(t+1)*D], s.Pre[t*D:(t+1)*D], w.dPre[t*D:(t+1)*D]
		mean, rstd := s.Mean[t], s.Rstd[t]
		var sumG, sumGX float64
		for j := range D {
			xhat := (x[j] - mean) * rstd
			dxh := dy[j] * gamma[j]
			gw[j] += dy[j] * xhat
			gb[j] += dy[j]
			sumG += float64(dxh)
			sumGX += float64(dxh * xhat)
		}
		mg, mgx := float32(sumG/float64(D)), float32(sumGX/float64(D))
		for j := range D {
			xhat := (x[j] - mean) * rstd
			dx[j] = rstd * (dy[j]*gamma[j] - mg - xhat*mgx)
		}
	}
}

func (m *Model) contextBackward(s *State, w *work, g *Grads) {
	c := m.Cfg
	R, Tc, L := c.CtxRows(), s.Tc, s.L
	CC, CD, NH := c.CtxChannels, c.CtxDim, c.CtxHeads
	hd := CD / NH
	n := R * Tc

	// Pooled[t, h·hd+j] = Σ_r Att[t,h,r]·F[r,t,h·hd+j].
	w.dF = grow(w.dF, n*CD)
	clear(w.dF)
	w.dS = grow(w.dS, n*NH)
	clear(w.dS)
	dPrior := g.of(m.RowPrior)
	w.dAtt = grow(w.dAtt, R)
	for t := range L {
		for h := range NH {
			a := s.Att[(t*NH+h)*R : (t*NH+h+1)*R]
			dp := w.dPooled[t*CD+h*hd : t*CD+(h+1)*hd]
			var dot float32
			for r := range R {
				f := s.CtxF[(r*Tc+t)*CD+h*hd : (r*Tc+t)*CD+(h+1)*hd]
				df := w.dF[(r*Tc+t)*CD+h*hd : (r*Tc+t)*CD+(h+1)*hd]
				var da float32
				for j := range dp {
					da += dp[j] * f[j]
					df[j] += a[r] * dp[j]
				}
				w.dAtt[r] = da
				dot += a[r] * da
			}
			// Softmax over the rows.
			for r := range R {
				ds := a[r] * (w.dAtt[r] - dot)
				dPrior[h*R+r] += ds
				w.dS[(r*Tc+t)*NH+h] = ds
			}
		}
	}

	// Scores = F·Scoreᵀ + b.
	weightGrad(g.of(m.CtxScore), g.of(m.CtxScoreBias), w.dS, s.CtxF, n, CD, NH)
	linalg.MatMul(w.dF, w.dS, m.CtxScore.W, n, NH, CD, false, false, true)

	// F = HardSwish(B), B = CtxD·PWᵀ + b.
	w.dB = grow(w.dB, n*CD)
	hardSwishGrad(w.dB, w.dF, s.CtxB)
	weightGrad(g.of(m.CtxPW), g.of(m.CtxPWBias), w.dB, s.CtxD, n, CC, CD)
	w.dCD = grow(w.dCD, n*CC)
	linalg.MatMul(w.dCD, w.dB, m.CtxPW.W, n, CD, CC, false, false, false)

	// CtxD = depthwise3x3(CtxH).
	w.dCH = grow(w.dCH, n*CC)
	clear(w.dCH)
	depthwise3x3Grad(w.dCH, g.of(m.CtxDW), w.dCD, s.CtxH, m.CtxDW.W, R, Tc, CC)

	// CtxH = HardSwish(CtxA), CtxA = CtxIn·Patchᵀ + b.
	hardSwishGrad(w.dCH, w.dCH, s.CtxA)
	weightGrad(g.of(m.CtxPatch), g.of(m.CtxPatchBias), w.dCH, s.CtxIn, n, c.Stride*c.Stride, CC)
}

// depthwise3x3Grad accumulates the gradients of depthwise3x3: dx and the
// kernel's dw [C, 3, 3]. w is the kernel in its stored layout [C, 1, 3, 3].
func depthwise3x3Grad(dx, dw, dy, x, w []float32, R, T, C int) {
	for r := range R {
		for t := range T {
			d := dy[(r*T+t)*C : (r*T+t+1)*C]
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
					in := x[(rr*T+tt)*C : (rr*T+tt+1)*C]
					out := dx[(rr*T+tt)*C : (rr*T+tt+1)*C]
					for ch := range C {
						dw[ch*9+k] += d[ch] * in[ch]
						out[ch] += w[ch*9+k] * d[ch]
					}
				}
			}
		}
	}
}

// Dense pairs the parameters with their gradients in g, for optim.AdamW:
// weight decay on the matrices and convolution kernels, not on the biases,
// the normalization and the row prior.
func (m *Model) Dense(g *Grads) []optim.Dense {
	ps := m.Params()
	out := make([]optim.Dense, len(ps))
	for i, p := range ps {
		out[i] = optim.Dense{W: p.W, G: g.G[i], Decay: len(p.Shape) > 1 && p != m.RowPrior}
	}
	return out
}
