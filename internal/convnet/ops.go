// Package convnet holds the 2D convolution operators of indecis's image
// models (the OCR line detector), forward and backward, written by hand
// like the rest of the library.
//
// Feature maps are channels-last, [H, W, C] row-major: a pointwise
// convolution is then a matrix product, and a depthwise one runs over
// contiguous channels. Kernels keep PyTorch's layout, [Cout, Cin, kh, kw].
package convnet

import (
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// Shape is the size of a feature map.
type Shape struct{ H, W, C int }

// Len is the number of values of a map of this shape.
func (s Shape) Len() int { return s.H * s.W * s.C }

// OutSize is the output size of a convolution along one axis.
func OutSize(n, k, stride, pad int) int { return (n+2*pad-k)/stride + 1 }

// Conv is a dense 2D convolution: kernel [Cout, Cin, K, K], stride, zero
// padding.
type Conv struct {
	Cin, Cout, K, Stride, Pad int
}

// Out returns the output shape for an input shape.
func (c Conv) Out(in Shape) Shape {
	return Shape{OutSize(in.H, c.K, c.Stride, c.Pad), OutSize(in.W, c.K, c.Stride, c.Pad), c.Cout}
}

// im2col writes the patches of x as rows [Ho·Wo, Cin·K·K], ordered as the
// kernel's (cin, ky, kx).
func (c Conv) im2col(col, x []float32, in Shape) {
	out := c.Out(in)
	k := c.Cin * c.K * c.K
	for oy := range out.H {
		for ox := range out.W {
			row := col[(oy*out.W+ox)*k : (oy*out.W+ox+1)*k]
			for ky := range c.K {
				y := oy*c.Stride + ky - c.Pad
				for kx := range c.K {
					x0 := ox*c.Stride + kx - c.Pad
					inside := y >= 0 && y < in.H && x0 >= 0 && x0 < in.W
					for ci := range c.Cin {
						v := float32(0)
						if inside {
							v = x[(y*in.W+x0)*in.C+ci]
						}
						row[ci*c.K*c.K+ky*c.K+kx] = v
					}
				}
			}
		}
	}
}

// Forward computes y = conv(x) + bias. col is a scratch buffer of
// Ho·Wo·Cin·K·K values, kept for Backward.
func (c Conv) Forward(y, col, x, w, bias []float32, in Shape) {
	out := c.Out(in)
	n, k := out.H*out.W, c.Cin*c.K*c.K
	c.im2col(col, x, in)
	for i := range n {
		copy(y[i*c.Cout:(i+1)*c.Cout], bias)
	}
	linalg.MatMul(y, col, w, n, k, c.Cout, false, true, true)
}

// Backward accumulates dw and dbias, and writes dx when not nil. col is
// Forward's.
func (c Conv) Backward(dx, dw, dBias, dy, col, w []float32, in Shape) {
	out := c.Out(in)
	n, k := out.H*out.W, c.Cin*c.K*c.K
	linalg.MatMul(dw, dy, col, c.Cout, n, k, true, false, true)
	for i := range n {
		linalg.AddTo(dBias, dy[i*c.Cout:(i+1)*c.Cout])
	}
	if dx == nil {
		return
	}
	dcol := make([]float32, n*k)
	linalg.MatMul(dcol, dy, w, n, c.Cout, k, false, false, false)
	clear(dx[:in.Len()])
	for oy := range out.H {
		for ox := range out.W {
			row := dcol[(oy*out.W+ox)*k : (oy*out.W+ox+1)*k]
			for ky := range c.K {
				y := oy*c.Stride + ky - c.Pad
				if y < 0 || y >= in.H {
					continue
				}
				for kx := range c.K {
					x0 := ox*c.Stride + kx - c.Pad
					if x0 < 0 || x0 >= in.W {
						continue
					}
					for ci := range c.Cin {
						dx[(y*in.W+x0)*in.C+ci] += row[ci*c.K*c.K+ky*c.K+kx]
					}
				}
			}
		}
	}
}

// Depthwise is a depthwise 2D convolution: kernel [C, 1, K, K], stride,
// padding K/2.
type Depthwise struct {
	C, K, Stride int
}

// Out returns the output shape for an input shape.
func (d Depthwise) Out(in Shape) Shape {
	p := d.K / 2
	return Shape{OutSize(in.H, d.K, d.Stride, p), OutSize(in.W, d.K, d.Stride, p), d.C}
}

// Forward computes y = depthwise(x) + bias (bias may be nil).
func (d Depthwise) Forward(y, x, w, bias []float32, in Shape) {
	out := d.Out(in)
	wt := transpose(w, d.C, d.K*d.K)
	p := d.K / 2
	for oy := range out.H {
		for ox := range out.W {
			o := y[(oy*out.W+ox)*d.C : (oy*out.W+ox+1)*d.C]
			if bias != nil {
				copy(o, bias)
			} else {
				clear(o)
			}
			for ky := range d.K {
				iy := oy*d.Stride + ky - p
				if iy < 0 || iy >= in.H {
					continue
				}
				for kx := range d.K {
					ix := ox*d.Stride + kx - p
					if ix < 0 || ix >= in.W {
						continue
					}
					t := ky*d.K + kx
					linalg.MulAddTo(o, wt[t*d.C:(t+1)*d.C], x[(iy*in.W+ix)*d.C:(iy*in.W+ix+1)*d.C])
				}
			}
		}
	}
}

// Backward accumulates dw (and dbias when not nil) and dx (not nil).
func (d Depthwise) Backward(dx, dw, dBias, dy, x, w []float32, in Shape) {
	out := d.Out(in)
	p := d.K / 2
	KK := d.K * d.K
	for oy := range out.H {
		for ox := range out.W {
			g := dy[(oy*out.W+ox)*d.C : (oy*out.W+ox+1)*d.C]
			if dBias != nil {
				linalg.AddTo(dBias, g)
			}
			for ky := range d.K {
				iy := oy*d.Stride + ky - p
				if iy < 0 || iy >= in.H {
					continue
				}
				for kx := range d.K {
					ix := ox*d.Stride + kx - p
					if ix < 0 || ix >= in.W {
						continue
					}
					t := ky*d.K + kx
					xi := x[(iy*in.W+ix)*d.C : (iy*in.W+ix+1)*d.C]
					di := dx[(iy*in.W+ix)*d.C : (iy*in.W+ix+1)*d.C]
					for c := range d.C {
						dw[c*KK+t] += g[c] * xi[c]
						di[c] += w[c*KK+t] * g[c]
					}
				}
			}
		}
	}
}

func transpose(w []float32, rows, cols int) []float32 {
	t := make([]float32, rows*cols)
	for r := range rows {
		for c := range cols {
			t[c*rows+r] = w[r*cols+c]
		}
	}
	return t
}

// Upsample2 doubles a map's height and width, nearest neighbour.
func Upsample2(y, x []float32, in Shape) {
	W2 := 2 * in.W
	for yy := range 2 * in.H {
		for xx := range W2 {
			copy(y[(yy*W2+xx)*in.C:(yy*W2+xx+1)*in.C], x[((yy/2)*in.W+xx/2)*in.C:((yy/2)*in.W+xx/2+1)*in.C])
		}
	}
}

// Upsample2Backward adds to dx the gradient of Upsample2.
func Upsample2Backward(dx, dy []float32, in Shape) {
	W2 := 2 * in.W
	for yy := range 2 * in.H {
		for xx := range W2 {
			linalg.AddTo(dx[((yy/2)*in.W+xx/2)*in.C:((yy/2)*in.W+xx/2+1)*in.C], dy[(yy*W2+xx)*in.C:(yy*W2+xx+1)*in.C])
		}
	}
}

// DepthToSpace rearranges [H, W, C·r·r] into [H·r, W·r, C], as ONNX's
// DepthToSpace in DCR mode: input channel (i·r + j)·C + c goes to output
// pixel (h·r + i, w·r + j), channel c.
func DepthToSpace(y, x []float32, in Shape, r int) {
	C := in.C / (r * r)
	Wo := in.W * r
	for h := range in.H {
		for w := range in.W {
			px := x[(h*in.W+w)*in.C : (h*in.W+w+1)*in.C]
			for i := range r {
				for j := range r {
					o := ((h*r+i)*Wo + w*r + j) * C
					copy(y[o:o+C], px[(i*r+j)*C:(i*r+j+1)*C])
				}
			}
		}
	}
}

// DepthToSpaceBackward writes dx, the gradient of DepthToSpace.
func DepthToSpaceBackward(dx, dy []float32, in Shape, r int) {
	C := in.C / (r * r)
	Wo := in.W * r
	for h := range in.H {
		for w := range in.W {
			px := dx[(h*in.W+w)*in.C : (h*in.W+w+1)*in.C]
			for i := range r {
				for j := range r {
					o := ((h*r+i)*Wo + w*r + j) * C
					copy(px[(i*r+j)*C:(i*r+j+1)*C], dy[o:o+C])
				}
			}
		}
	}
}

// Softplus writes y = log(1 + e^x), stable for large x.
func Softplus(y, x []float32) {
	linalg.ZeroUpper()
	for i, v := range x[:len(y)] {
		if v > 20 {
			y[i] = v
		} else {
			y[i] = float32(math.Log1p(math.Exp(float64(v))))
		}
	}
}

// SoftplusBackward writes dx = dy·σ(x).
func SoftplusBackward(dx, dy, x []float32) {
	linalg.ZeroUpper()
	for i, v := range x[:len(dx)] {
		dx[i] = dy[i] / float32(1+math.Exp(-float64(v)))
	}
}
