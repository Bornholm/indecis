package linedet

import (
	"github.com/bornholm/indecis/internal/convnet"
	"github.com/bornholm/indecis/linalg"
)

// The detector is a graph of some forty operations. Rather than a forward
// and a backward written by hand for each, as linerec does, the forward
// records each operation on a tape with its backward step; Backward runs
// the tape in reverse. Buffers are reused from page to page: the i-th
// operation of a page gets the i-th buffer of the previous one.

// Tensor is a feature map and, during Backward, its gradient.
type Tensor struct {
	V, G []float32
	S    convnet.Shape
}

// Tape records a forward pass. A Tape is reused from page to page.
type Tape struct {
	steps []func(g *Grads)
	pool  []*Tensor
	used  int
	cols  [][]float32 // im2col buffers of the dense convolutions
	ncols int
	wt    []float32 // a depthwise kernel transposed, see convnet.Depthwise
	dcol  []float32 // a dense convolution's col gradient
	train bool      // record the backward steps
}

func (t *Tape) reset(train bool) {
	t.steps = t.steps[:0]
	t.used, t.ncols = 0, 0
	t.train = train
}

func (t *Tape) tensor(s convnet.Shape) *Tensor {
	if t.used == len(t.pool) {
		t.pool = append(t.pool, &Tensor{})
	}
	x := t.pool[t.used]
	t.used++
	x.S = s
	x.V = grow(x.V, s.Len())
	x.G = x.G[:0]
	return x
}

func (t *Tape) col(n int) []float32 {
	if t.ncols == len(t.cols) {
		t.cols = append(t.cols, nil)
	}
	c := grow(t.cols[t.ncols], n)
	t.cols[t.ncols] = c
	t.ncols++
	return c
}

func (t *Tape) record(f func(g *Grads)) {
	if t.train {
		t.steps = append(t.steps, f)
	}
}

// grad returns x's gradient buffer, zeroed the first time.
func grad(x *Tensor) []float32 {
	if len(x.G) != len(x.V) {
		x.G = grow(x.G, len(x.V))
		clear(x.G)
	}
	return x.G
}

// pw is a pointwise convolution: y = x·Wᵀ + b, W [out, in, 1, 1].
func (t *Tape) pw(x *Tensor, w, b *Param) *Tensor {
	out := w.Shape[0]
	n := x.S.H * x.S.W
	y := t.tensor(convnet.Shape{H: x.S.H, W: x.S.W, C: out})
	for i := range n {
		copy(y.V[i*out:(i+1)*out], b.W)
	}
	linalg.MatMul(y.V, x.V, w.W, n, x.S.C, out, false, true, true)
	t.record(func(g *Grads) {
		dw, db := g.of(w), g.of(b)
		linalg.MatMul(dw, y.G, x.V, out, n, x.S.C, true, false, true)
		for i := range n {
			linalg.AddTo(db, y.G[i*out:(i+1)*out])
		}
		linalg.MatMul(grad(x), y.G, w.W, n, out, x.S.C, false, false, true)
	})
	return y
}

func (t *Tape) dw(x *Tensor, w, b *Param, k, stride int) *Tensor {
	op := convnet.Depthwise{C: x.S.C, K: k, Stride: stride}
	y := t.tensor(op.Out(x.S))
	var bias []float32
	if b != nil {
		bias = b.W
	}
	t.wt = grow(t.wt, len(w.W))
	op.Forward(y.V, x.V, w.W, bias, t.wt, x.S)
	t.record(func(g *Grads) {
		var db []float32
		if b != nil {
			db = g.of(b)
		}
		op.Backward(grad(x), g.of(w), db, y.G, x.V, w.W, x.S)
	})
	return y
}

// conv is a dense convolution; dx says whether x needs its gradient (not
// for the page itself).
func (t *Tape) conv(x *Tensor, w, b *Param, k, stride, pad int, dx bool) *Tensor {
	op := convnet.Conv{Cin: x.S.C, Cout: w.Shape[0], K: k, Stride: stride, Pad: pad}
	out := op.Out(x.S)
	y := t.tensor(out)
	col := t.col(out.H * out.W * op.Cin * k * k)
	op.Forward(y.V, col, x.V, w.W, b.W, x.S)
	t.record(func(g *Grads) {
		var d, dcol []float32
		if dx {
			d = make([]float32, len(x.V))
			t.dcol = grow(t.dcol, len(col))
			dcol = t.dcol
		}
		op.Backward(d, g.of(w), g.of(b), y.G, col, dcol, w.W, x.S)
		if dx {
			linalg.AddTo(grad(x), d)
		}
	})
	return y
}

func (t *Tape) hs(x *Tensor) *Tensor {
	y := t.tensor(x.S)
	linalg.HardSwish(y.V, x.V)
	t.record(func(g *Grads) {
		dx := grad(x)
		for i, v := range x.V {
			switch {
			case v <= -3:
			case v >= 3:
				dx[i] += y.G[i]
			default:
				dx[i] += y.G[i] * (v/3 + 0.5)
			}
		}
	})
	return y
}

func (t *Tape) add(a, b *Tensor) *Tensor {
	y := t.tensor(a.S)
	copy(y.V, a.V)
	linalg.AddTo(y.V, b.V)
	t.record(func(g *Grads) {
		linalg.AddTo(grad(a), y.G)
		linalg.AddTo(grad(b), y.G)
	})
	return y
}

func (t *Tape) up2(x *Tensor) *Tensor {
	y := t.tensor(convnet.Shape{H: 2 * x.S.H, W: 2 * x.S.W, C: x.S.C})
	convnet.Upsample2(y.V, x.V, x.S)
	t.record(func(g *Grads) { convnet.Upsample2Backward(grad(x), y.G, x.S) })
	return y
}

// s2d moves 2×2 blocks of pixels into channels (DepthToSpace's inverse).
func (t *Tape) s2d(x *Tensor) *Tensor {
	out := convnet.Shape{H: x.S.H / 2, W: x.S.W / 2, C: 4 * x.S.C}
	y := t.tensor(out)
	convnet.DepthToSpaceBackward(y.V, x.V, out, 2)
	t.record(func(g *Grads) {
		d := make([]float32, len(x.V))
		convnet.DepthToSpace(d, y.G, out, 2)
		linalg.AddTo(grad(x), d)
	})
	return y
}

func (t *Tape) d2s(x *Tensor) *Tensor {
	y := t.tensor(convnet.Shape{H: 2 * x.S.H, W: 2 * x.S.W, C: x.S.C / 4})
	convnet.DepthToSpace(y.V, x.V, x.S, 2)
	t.record(func(g *Grads) {
		d := make([]float32, len(x.V))
		convnet.DepthToSpaceBackward(d, y.G, x.S, 2)
		linalg.AddTo(grad(x), d)
	})
	return y
}

// concat joins two maps of the same size along the channels.
func (t *Tape) concat(a, b *Tensor) *Tensor {
	C := a.S.C + b.S.C
	y := t.tensor(convnet.Shape{H: a.S.H, W: a.S.W, C: C})
	n := a.S.H * a.S.W
	for i := range n {
		copy(y.V[i*C:i*C+a.S.C], a.V[i*a.S.C:(i+1)*a.S.C])
		copy(y.V[i*C+a.S.C:(i+1)*C], b.V[i*b.S.C:(i+1)*b.S.C])
	}
	t.record(func(g *Grads) {
		ga, gb := grad(a), grad(b)
		for i := range n {
			linalg.AddTo(ga[i*a.S.C:(i+1)*a.S.C], y.G[i*C:i*C+a.S.C])
			linalg.AddTo(gb[i*b.S.C:(i+1)*b.S.C], y.G[i*C+a.S.C:(i+1)*C])
		}
	})
	return y
}

func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}
