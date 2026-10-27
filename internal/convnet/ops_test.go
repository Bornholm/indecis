package convnet

import (
	"math"
	"math/rand"
	"testing"
)

func randn(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// checkGrad compares an analytic gradient with central differences of
// L = <f(), g>. The operators are linear in each input, so the
// differences are exact up to rounding.
func checkGrad(t *testing.T, name string, x, grad []float32, loss func() float64) {
	t.Helper()
	const h = 0.25
	for _, i := range []int{0, len(x) / 3, len(x) / 2, len(x) - 1} {
		v := x[i]
		x[i] = v + h
		lp := loss()
		x[i] = v - h
		lm := loss()
		x[i] = v
		fd := (lp - lm) / (2 * h)
		if math.Abs(fd-float64(grad[i])) > 1e-3*max(1, math.Abs(fd)) {
			t.Errorf("%s[%d]: gradient %.5f, finite difference %.5f", name, i, grad[i], fd)
		}
	}
}

func TestConv(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, c := range []Conv{{Cin: 1, Cout: 4, K: 3, Stride: 2, Pad: 1}, {Cin: 3, Cout: 5, K: 3, Stride: 1, Pad: 1}, {Cin: 2, Cout: 3, K: 1, Stride: 1}} {
		in := Shape{7, 9, c.Cin}
		out := c.Out(in)
		x, w, b := randn(r, in.Len()), randn(r, c.Cout*c.Cin*c.K*c.K), randn(r, c.Cout)
		y := make([]float32, out.Len())
		col := make([]float32, out.H*out.W*c.Cin*c.K*c.K)
		c.Forward(y, col, x, w, b, in)
		// Direct definition.
		for oy := range out.H {
			for ox := range out.W {
				for co := range c.Cout {
					s := float64(b[co])
					for ci := range c.Cin {
						for ky := range c.K {
							for kx := range c.K {
								iy, ix := oy*c.Stride+ky-c.Pad, ox*c.Stride+kx-c.Pad
								if iy >= 0 && iy < in.H && ix >= 0 && ix < in.W {
									s += float64(w[((co*c.Cin+ci)*c.K+ky)*c.K+kx]) * float64(x[(iy*in.W+ix)*in.C+ci])
								}
							}
						}
					}
					if got := y[(oy*out.W+ox)*c.Cout+co]; math.Abs(float64(got)-s) > 1e-4 {
						t.Fatalf("%+v: y[%d,%d,%d] = %v, want %v", c, oy, ox, co, got, s)
					}
				}
			}
		}
		g := randn(r, out.Len())
		dx, dw, db := make([]float32, in.Len()), make([]float32, len(w)), make([]float32, len(b))
		c.Backward(dx, dw, db, g, col, w, in)
		loss := func() float64 {
			c.Forward(y, col, x, w, b, in)
			return dot(y, g)
		}
		checkGrad(t, "conv x", x, dx, loss)
		checkGrad(t, "conv w", w, dw, loss)
		checkGrad(t, "conv b", b, db, loss)
	}
}

func TestDepthwise(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, d := range []Depthwise{{C: 5, K: 3, Stride: 1}, {C: 4, K: 5, Stride: 2}, {C: 3, K: 3, Stride: 2}} {
		in := Shape{8, 11, d.C}
		out := d.Out(in)
		x, w, b := randn(r, in.Len()), randn(r, d.C*d.K*d.K), randn(r, d.C)
		y := make([]float32, out.Len())
		d.Forward(y, x, w, b, in)
		p := d.K / 2
		for oy := range out.H {
			for ox := range out.W {
				for c := range d.C {
					s := float64(b[c])
					for ky := range d.K {
						for kx := range d.K {
							iy, ix := oy*d.Stride+ky-p, ox*d.Stride+kx-p
							if iy >= 0 && iy < in.H && ix >= 0 && ix < in.W {
								s += float64(w[c*d.K*d.K+ky*d.K+kx]) * float64(x[(iy*in.W+ix)*d.C+c])
							}
						}
					}
					if got := y[(oy*out.W+ox)*d.C+c]; math.Abs(float64(got)-s) > 1e-4 {
						t.Fatalf("%+v: y = %v, want %v", d, got, s)
					}
				}
			}
		}
		g := randn(r, out.Len())
		dx, dw, db := make([]float32, in.Len()), make([]float32, len(w)), make([]float32, len(b))
		d.Backward(dx, dw, db, g, x, w, in)
		loss := func() float64 {
			d.Forward(y, x, w, b, in)
			return dot(y, g)
		}
		checkGrad(t, "depthwise x", x, dx, loss)
		checkGrad(t, "depthwise w", w, dw, loss)
		checkGrad(t, "depthwise b", b, db, loss)
	}
}

func TestReshapes(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	in := Shape{3, 4, 8}
	x := randn(r, in.Len())

	up := make([]float32, 4*in.Len())
	Upsample2(up, x, in)
	if up[((2*1+1)*8+(2*2+1))*8+5] != x[(1*4+2)*8+5] {
		t.Fatal("Upsample2: wrong source pixel")
	}
	g := randn(r, len(up))
	dx := make([]float32, len(x))
	Upsample2Backward(dx, g, in)
	checkGrad(t, "upsample", x, dx, func() float64 { Upsample2(up, x, in); return dot(up, g) })

	ds := make([]float32, in.Len())
	DepthToSpace(ds, x, in, 2)
	// Output pixel (h·2+i, w·2+j), channel c ← input (h, w), channel (i·2+j)·2+c.
	if ds[((1*2+1)*8+(2*2+0))*2+1] != x[(1*4+2)*8+(1*2+0)*2+1] {
		t.Fatal("DepthToSpace: wrong mapping")
	}
	g = randn(r, len(ds))
	DepthToSpaceBackward(dx, g, in, 2)
	checkGrad(t, "depth to space", x, dx, func() float64 { DepthToSpace(ds, x, in, 2); return dot(ds, g) })

	sp := make([]float32, len(x))
	Softplus(sp, x)
	for i, v := range x {
		if want := math.Log1p(math.Exp(float64(v))); math.Abs(float64(sp[i])-want) > 1e-6 {
			t.Fatalf("Softplus(%v) = %v, want %v", v, sp[i], want)
		}
	}
	g = randn(r, len(x))
	SoftplusBackward(dx, g, x)
	for i, v := range x {
		if want := float64(g[i]) / (1 + math.Exp(-float64(v))); math.Abs(float64(dx[i])-want) > 1e-6 {
			t.Fatalf("SoftplusBackward: %v, want %v", dx[i], want)
		}
	}
}
