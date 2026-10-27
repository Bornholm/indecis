package linedet

import (
	"math"
	"math/rand"
	"testing"
)

func tiny() Config {
	return Config{Stem: 4, StemLat: 2, FPN: 6, HeightScale: 8,
		Stages: []Stage{{5, 3, 1, 8}, {6, 3, 1, 8}, {7, 5, 1, 9}, {8, 5, 1, 10}}}
}

// TestBackwardFiniteDifferences checks every parameter's gradient on a
// few weights, for L = <maps, g>.
func TestBackwardFiniteDifferences(t *testing.T) {
	m := New(tiny(), 1)
	r := rand.New(rand.NewSource(2))
	H, W := 64, 96
	ink := make([]float32, H*W)
	for i := range ink {
		ink[i] = r.Float32()
	}
	var tp Tape
	out := m.Forward(&tp, ink, H, W, true)
	if out.S.H != H/2 || out.S.W != W/2 || out.S.C != 3 {
		t.Fatalf("output shape %+v", out.S)
	}
	gOut := make([]float32, len(out.V))
	for i := range gOut {
		gOut[i] = float32(r.NormFloat64())
	}
	g := m.NewGrads()
	m.Backward(&tp, out, gOut, g)
	loss := func() float64 {
		var t2 Tape
		o := m.Forward(&t2, ink, H, W, false)
		var s float64
		for i, v := range o.V {
			s += float64(v) * float64(gOut[i])
		}
		return s
	}
	const h = 1e-2
	for i, p := range m.Params() {
		for _, j := range []int{0, len(p.W) / 2, len(p.W) - 1} {
			v := p.W[j]
			p.W[j] = v + h
			lp := loss()
			p.W[j] = v - h
			lm := loss()
			p.W[j] = v
			fd := (lp - lm) / (2 * h)
			if got := float64(g.G[i][j]); math.Abs(fd-got) > 2e-2*max(1, math.Abs(fd)) {
				t.Errorf("%s[%d]: gradient %.5f, finite difference %.5f", p.Name, j, got, fd)
			}
		}
	}
}

func TestBaseSize(t *testing.T) {
	n := 0
	for _, p := range New(Base(), 1).Params() {
		n += len(p.W)
	}
	// Indra's det.onnx: 90,568 (its batch norms folded into the kernels).
	if n < 85000 || n > 95000 {
		t.Fatalf("%d parameters", n)
	}
	t.Logf("%d parameters", n)
}
