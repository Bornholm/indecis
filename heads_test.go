package indecis

import (
	"math"
	"math/rand"
	"testing"
)

// Checks the gradients of the three heads by finite differences, in
// float64 on the loss side.
func TestHeadGradients(t *testing.T) {
	const H = 6
	r := rand.New(rand.NewSource(1))
	cases := []struct {
		q      Question
		target []float64
	}{
		{NewNoul("n", ""), []float64{1}},
		{NewNoul("soft", ""), []float64{0.3}},
		{NewChoice("c", "", "a", "b", "c", "d"), []float64{0, 0, 1, 0}},
		{NewChoice("csoft", "", "a", "b", "c"), []float64{0.2, 0.5, 0.3}},
		{NewScore("s", "", "0", "1", "2", "3"), []float64{2}},
	}
	for _, c := range cases {
		t.Run(c.q.Name, func(t *testing.T) {
			h := newHead(c.q, H, r)
			for i := range h.w {
				h.w[i] = float32(r.NormFloat64())
			}
			h.enableGrad()
			x := make([]float32, H)
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
			loss := func() float64 {
				l, _ := h.lossGrad(h.logits(x), c.target)
				return l
			}
			_, dz := h.lossGrad(h.logits(x), c.target)
			dx := make([]float32, H)
			h.backward(x, dz, dx)

			const eps = 1e-3
			fd := func(p []float32, i int) float64 {
				o := p[i]
				p[i] = o + eps
				lp := loss()
				p[i] = o - eps
				lm := loss()
				p[i] = o
				return (lp - lm) / (2 * eps)
			}
			for i := range h.w {
				if d := fd(h.w, i); math.Abs(d-float64(h.gw[i])) > 1e-3+1e-2*math.Abs(d) {
					t.Errorf("w[%d]: %v vs %v", i, h.gw[i], d)
				}
			}
			for i := range h.b {
				if d := fd(h.b, i); math.Abs(d-float64(h.gb[i])) > 1e-3+1e-2*math.Abs(d) {
					t.Errorf("b[%d]: %v vs %v", i, h.gb[i], d)
				}
			}
			for i := range x {
				if d := fd(x, i); math.Abs(d-float64(dx[i])) > 1e-3+1e-2*math.Abs(d) {
					t.Errorf("x[%d]: %v vs %v", i, dx[i], d)
				}
			}
		})
	}
}

func TestScoreAnswerIsADistribution(t *testing.T) {
	h := newHead(NewScore("s", "", "bas", "moyen", "haut"), 4, rand.New(rand.NewSource(1)))
	// Thresholds reversed on purpose: the answer must remain a distribution.
	a := h.answer([]float64{-1, 2}, 1)
	var sum float64
	for _, p := range a.Probs {
		if p < 0 {
			t.Fatalf("negative probability: %v", a.Probs)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 || a.Score < 0 || a.Score > 2 {
		t.Fatalf("got %+v", a)
	}
}

func TestTargets(t *testing.T) {
	c := NewChoice("c", "", "a", "b")
	if tg, ok, err := c.target(map[string]any{"a": 1.0, "b": 3.0}); !ok || err != nil || tg[1] != 0.75 {
		t.Fatalf("distribution: %v %v %v", tg, ok, err)
	}
	if _, _, err := c.target("z"); err == nil {
		t.Fatal("unknown option accepted")
	}
	s := NewScore("s", "", "bas", "haut")
	if tg, ok, _ := s.target("haut"); !ok || tg[0] != 1 {
		t.Fatalf("level by name: %v", tg)
	}
	if _, _, err := s.target(1.5); err == nil {
		t.Fatal("fractional level accepted")
	}
	if _, ok, err := NewNoul("n", "").target(nil); ok || err != nil {
		t.Fatal("missing label mishandled")
	}
}
