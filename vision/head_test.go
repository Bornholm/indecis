package vision

import (
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/bornholm/indecis"
)

// A spatial task: an "object" (a strong feature 0) sits in one patch of an
// 8×8 grid; the head must tell whether it is left or right of the center,
// and whether a second signal (feature 1) is present anywhere.
func TestHeadLearnsPositions(t *testing.T) {
	const T, H = 64, 16
	schema := indecis.Schema{
		indecis.NewChoice("side", "", "left", "right"),
		indecis.NewNoul("signal", ""),
	}
	r := rand.New(rand.NewSource(1))
	make1 := func() HeadExample {
		p := make([]float32, T*H)
		for i := range p {
			p[i] = float32(r.NormFloat64() * 0.3)
		}
		pos := r.Intn(T)
		p[pos*H] += 3
		side := "left"
		if pos%8 >= 4 {
			side = "right"
		}
		signal := r.Intn(2) == 0
		if signal {
			p[r.Intn(T)*H+1] += 3
		}
		return HeadExample{Patches: ToBF16(p), Labels: map[string]any{"side": side, "signal": signal}}
	}
	var train, test []HeadExample
	for i := 0; i < 3000; i++ {
		train = append(train, make1())
	}
	for i := 0; i < 500; i++ {
		test = append(test, make1())
	}
	h, err := NewHead(schema, T, H, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultHeadTrainOptions()
	opts.Epochs = 15
	if err := h.Fit(train, opts); err != nil {
		t.Fatal(err)
	}
	acc, err := h.Evaluate(test)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("accuracy: side %.3f, signal %.3f", acc["side"], acc["signal"])
	if acc["side"] < 0.95 || acc["signal"] < 0.95 {
		t.Fatalf("accuracy %v", acc)
	}

	// Same data, same seed: the same head, bit for bit.
	h2, _ := NewHead(schema, T, H, 8, 1)
	if err := h2.Fit(train, opts); err != nil {
		t.Fatal(err)
	}
	for i := range h.W1 {
		if h.W1[i] != h2.W1[i] {
			t.Fatalf("training not reproducible at W1[%d]", i)
		}
	}
}

func TestFitRefusesNoExamples(t *testing.T) {
	h, err := NewHead(indecis.Schema{indecis.NewNoul("signal", "")}, 4, 2, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Fit(nil, DefaultHeadTrainOptions()); err == nil {
		t.Fatal("Fit without examples: no error")
	}
}

// Saving over a model replaces it whole and leaves no temporary file.
func TestSaveHeadOverwrites(t *testing.T) {
	dir := t.TempDir()
	schema := indecis.Schema{indecis.NewNoul("signal", "")}
	for _, k := range []int{2, 3} {
		h, err := NewHead(schema, 4, 2, k, 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := SaveHead(dir, "backbone", h); err != nil {
			t.Fatal(err)
		}
		m, err := readManifest(dir)
		if err != nil || m == nil {
			t.Fatalf("manifest after save: %v, %v", m, err)
		}
		got, err := loadHead(dir, m)
		if err != nil {
			t.Fatal(err)
		}
		if got.K != k {
			t.Errorf("K = %d after saving K = %d", got.K, k)
		}
	}
	tmp, _ := filepath.Glob(filepath.Join(dir, "*.tmp"))
	if len(tmp) > 0 {
		t.Errorf("temporary files left: %v", tmp)
	}
}
