package linerec

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/bornholm/indecis/optim"
	"github.com/bornholm/indecis/safetensors"
)

// lossGrads runs the fixture lines through m and returns the summed CTC
// loss and the gradients.
func lossGrads(t *testing.T, m *Model, c tinyCases, lines [][]float32) (float64, *Grads) {
	g := m.NewGrads()
	var s State
	var total float64
	for i, w := range c.Widths {
		m.Forward(&s, lines[i], w)
		d := make([]float32, s.L*m.Cfg.Vocab)
		loss, ok := CTC(s.Logits, s.L, m.Cfg.Vocab, c.Targets[i], d)
		if !ok {
			t.Fatalf("line %d: no alignment", i)
		}
		total += loss
		m.Backward(&s, d, g)
	}
	return total, g
}

// TestBackwardMatchesPyTorch compares the loss and every parameter's
// gradient with PyTorch's (ctc_loss, sum over the lines).
func TestBackwardMatchesPyTorch(t *testing.T) {
	c, lines := loadTiny(t)
	dir := filepath.Join(tinyDir, "layer")
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	loss, g := lossGrads(t, m, c, lines)
	if d := math.Abs(loss - c.Layer.Loss); d > 1e-4*math.Abs(c.Layer.Loss) {
		t.Errorf("loss %.6f, PyTorch %.6f", loss, c.Layer.Loss)
	}
	want, _, err := safetensors.ReadFile(filepath.Join(dir, "grads.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range m.Params() {
		w, ok := want[p.Name]
		if !ok {
			t.Fatalf("no PyTorch gradient for %s", p.Name)
		}
		var num, den float64
		for j, v := range g.G[i] {
			d := float64(v - w.Data[j])
			num += d * d
			den += float64(w.Data[j]) * float64(w.Data[j])
		}
		// A floor of 1e-2 on the norm: the score bias's true gradient is 0
		// (a softmax ignores a constant added to all its inputs).
		rel := math.Sqrt(num / max(den, 1e-4))
		if rel > 1e-4 {
			t.Errorf("%s: relative error %.2e (|g| %.3e)", p.Name, rel, math.Sqrt(den))
		}
	}
}

// TestCTCFiniteDifferences checks the CTC gradient against finite
// differences of the loss.
func TestCTCFiniteDifferences(t *testing.T) {
	L, V := 7, 5
	logits := make([]float32, L*V)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*1.7)) * 2
	}
	target := []int{2, 2, 4}
	d := make([]float32, L*V)
	if _, ok := CTC(logits, L, V, target, d); !ok {
		t.Fatal("no alignment")
	}
	const h = 1e-2
	for i := range logits {
		x := logits[i]
		logits[i] = x + h
		lp, _ := CTC(logits, L, V, target, nil)
		logits[i] = x - h
		lm, _ := CTC(logits, L, V, target, nil)
		logits[i] = x
		fd := (lp - lm) / (2 * h)
		if math.Abs(fd-float64(d[i])) > 1e-3 {
			t.Fatalf("logit %d: gradient %.5f, finite difference %.5f", i, d[i], fd)
		}
	}
	// A repeated label needs a blank between: 3 symbols "2 2 4" need 4 steps.
	if _, ok := CTC(logits[:3*V], 3, V, target, nil); ok {
		t.Fatal("3 steps cannot hold 2 2 4")
	}
}

// TestTrainStepMatchesPyTorch: gradients, clipping and one AdamW step.
func TestTrainStepMatchesPyTorch(t *testing.T) {
	c, lines := loadTiny(t)
	dir := filepath.Join(tinyDir, "layer")
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, g := lossGrads(t, m, c, lines)
	dense := m.Dense(g)
	norm := optim.ClipGradNorm(c.Clip, dense, nil)
	if math.Abs(norm-c.Layer.TotalNorm) > 1e-4*c.Layer.TotalNorm {
		t.Errorf("gradient norm %.6f, PyTorch %.6f", norm, c.Layer.TotalNorm)
	}
	cfg := optim.DefaultConfig(c.LR)
	cfg.WeightDecay = c.WD
	optim.New(cfg).Step(c.LR, dense, nil)
	want, _, err := safetensors.ReadFile(filepath.Join(dir, "after.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range m.Params() {
		if p == m.CtxScoreBias {
			// Its gradient is rounding noise (see above), which Adam
			// normalizes to steps of ±lr: the two noises differ.
			continue
		}
		if d := maxDiff(p.W, want[p.Name].Data); d > 1e-6 {
			t.Errorf("%s: max |Δ| after the step %.2e", p.Name, d)
		}
	}
}

// TestBatchNormFiniteDifferences checks the backward pass of the batch
// norm variant (running statistics frozen) by finite differences on a few
// weights of each parameter.
func TestBatchNormFiniteDifferences(t *testing.T) {
	c, lines := loadTiny(t)
	m, err := Load(filepath.Join(tinyDir, "batch"))
	if err != nil {
		t.Fatal(err)
	}
	_, g := lossGrads(t, m, c, lines)
	const h = 1e-2
	for i, p := range m.Params() {
		for _, j := range []int{0, len(p.W) / 2, len(p.W) - 1} {
			x := p.W[j]
			p.W[j] = x + h
			lp, _ := lossGrads(t, m, c, lines)
			p.W[j] = x - h
			lm, _ := lossGrads(t, m, c, lines)
			p.W[j] = x
			fd := (lp - lm) / (2 * h)
			if got := float64(g.G[i][j]); math.Abs(fd-got) > 2e-2*max(1, math.Abs(fd)) {
				t.Errorf("%s[%d]: gradient %.5f, finite difference %.5f", p.Name, j, got, fd)
			}
		}
	}
}
