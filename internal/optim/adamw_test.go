package optim

import (
	"math"
	"testing"
)

// Values obtained with torch.optim.AdamW(lr=0.1, weight_decay=0.01) on
// w = [1, -2], g = [0.5, 0.25] then g = [-1, 0.1]:
//
//	w = torch.tensor([1., -2.], requires_grad=True)
//	opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.01)
//	for g in ([0.5, 0.25], [-1., 0.1]):
//	    w.grad = torch.tensor(g); opt.step()
func TestAdamWMatchesTorch(t *testing.T) {
	w := []float32{1, -2}
	g := []float32{0.5, 0.25}
	o := New(Config{LR: 0.1, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.01})
	o.Step(0.1, []Dense{{W: w, G: g, Decay: true}}, nil)
	// First Adam step: w <- w(1 - lr*wd) - lr*sign(g)
	want1 := []float64{1*(1-0.001) - 0.1, -2*(1-0.001) - 0.1}
	for i := range w {
		if math.Abs(float64(w[i])-want1[i]) > 1e-6 {
			t.Fatalf("step 1: w[%d] = %v, want %v", i, w[i], want1[i])
		}
	}
	g[0], g[1] = -1, 0.1
	o.Step(0.1, []Dense{{W: w, G: g, Decay: true}}, nil)
	// Output of the script above.
	want2 := []float64{0.9347114, -2.1857595} // output of torch 2.14
	for i := range w {
		if math.Abs(float64(w[i])-want2[i]) > 2e-6 {
			t.Fatalf("step 2: w[%d] = %v, want %v", i, w[i], want2[i])
		}
	}
}

func TestSparseRowsOnly(t *testing.T) {
	w := []float32{1, 1, 2, 2, 3, 3}
	o := New(DefaultConfig(0.1))
	o.Step(0.1, nil, []Sparse{{W: w, Width: 2, Rows: map[int32][]float32{1: {1, -1}}}})
	if w[0] != 1 || w[1] != 1 || w[4] != 3 || w[5] != 3 {
		t.Fatalf("absent rows modified: %v", w)
	}
	if math.Abs(float64(w[2])-1.9) > 1e-6 || math.Abs(float64(w[3])-2.1) > 1e-6 {
		t.Fatalf("row 1: %v", w[2:4])
	}
}

func TestClipGradNorm(t *testing.T) {
	d := []Dense{{G: []float32{3, 0}}}
	s := []Sparse{{Rows: map[int32][]float32{7: {0, 4}}}}
	if n := ClipGradNorm(1, d, s); math.Abs(n-5) > 1e-9 {
		t.Fatalf("norm %v", n)
	}
	if math.Abs(float64(d[0].G[0])-0.6) > 1e-5 || math.Abs(float64(s[0].Rows[7][1])-0.8) > 1e-5 {
		t.Fatalf("clipping: %v %v", d[0].G, s[0].Rows[7])
	}
}
