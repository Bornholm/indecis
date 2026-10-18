package modernbert

import "testing"

// A pruned model computes what the full model computes with the output
// weights of the removed neurons set to zero, in inference and in
// training; the activity of layer 0, upstream of any pruning, is that of
// the kept neurons.
func TestPruneMLP(t *testing.T) {
	keep := [][]int{{0, 2, 3, 5}, {1, 2, 4, 5}, {0, 1, 2, 3}, {5, 4, 0, 1}}
	full, pruned := tinyModel(t, 3), tinyModel(t, 3)
	seqs := [][]int32{{1, 4, 2, 9, 3, 16, 7}, {5, 5, 2}}
	before, err := full.MLPActivity(seqs)
	if err != nil {
		t.Fatal(err)
	}
	H, I := full.Cfg.Hidden, full.Cfg.Intermediate
	for l, L := range full.Layers {
		kept := map[int]bool{}
		for _, i := range keep[l] {
			kept[i] = true
		}
		for h := 0; h < H; h++ {
			for i := 0; i < I; i++ {
				if !kept[i] {
					L.WoMLP.W[h*I+i] = 0
				}
			}
		}
	}
	full.Invalidate()
	if err := pruned.PruneMLP(keep); err != nil {
		t.Fatal(err)
	}
	if pruned.Cfg.Intermediate != 4 {
		t.Fatalf("intermediate %d", pruned.Cfg.Intermediate)
	}
	want, _ := full.EncodeTokens(seqs)
	got, err := pruned.EncodeTokens(seqs)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(got, want); d > 1e-5 {
		t.Fatalf("inference: max difference %g", d)
	}
	b := NewBatch(seqs, 0)
	sf, _ := full.Forward(b)
	sp, err := pruned.Forward(b)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(sp.Hidden, sf.Hidden); d > 1e-5 {
		t.Fatalf("training: max difference %g", d)
	}
	after, _ := pruned.MLPActivity(seqs)
	for j, i := range keep[0] {
		if d := after[0][j] - before[0][i]; d > 1e-6 || d < -1e-6 {
			t.Fatalf("layer 0 neuron %d: activity %v, was %v", i, after[0][j], before[0][i])
		}
	}
	if err := pruned.PruneMLP([][]int{{0}, {0}, {0}, {0, 1}}); err == nil {
		t.Fatal("unequal layers accepted")
	}
}
