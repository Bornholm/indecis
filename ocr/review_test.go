package ocr

import (
	"context"
	"strings"
	"testing"

	"github.com/bornholm/indecis/internal/linedet"
	"github.com/bornholm/indecis/internal/linerec"
)

// TestTrainStopsOnUnusableLines: an endless source whose lines all fall
// outside the vocabulary ends the run with an error instead of drawing
// forever.
func TestTrainStopsOnUnusableLines(t *testing.T) {
	v, _ := NewVocab("abc")
	cfg := linerec.Base(len(v.Chars) + 1)
	cfg.Dim, cfg.Expand, cfg.SE, cfg.Blocks, cfg.CtxChannels, cfg.CtxDim = 16, 16, 4, 1, 4, 8
	m, err := linerec.New(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	lines := pageLines(t) // texts full of characters outside "abc"
	_, err = Train(context.Background(), m, v, NewMemorySource(lines, 1), TrainOptions{Steps: 3, Batch: 2, LR: 1e-3})
	if err == nil || !strings.Contains(err.Error(), "outside the vocabulary") {
		t.Fatalf("err = %v", err)
	}
}

func TestTrainDetectorOptions(t *testing.T) {
	m := linedet.New(linedet.Base(), 1)
	ps := []PageSample{{Image: pageDir + "/page.png", Boxes: [][4]float64{{60, 60, 1111, 100}}, Texts: []string{"x"}}}
	ctx := context.Background()
	// MinSide alone: MaxSide follows it.
	if _, err := TrainDetector(ctx, m, ps, DetTrainOptions{Steps: 1, Batch: 1, LR: 1e-3, Crop: 64, MinSide: 300}); err != nil {
		t.Fatal(err)
	}
	for _, o := range []DetTrainOptions{
		{Steps: 1, Batch: 1, MinSide: 900, MaxSide: 600},
		{Steps: 0, Batch: 1},
		{Steps: 1, Batch: 1, Crop: 16},
	} {
		if _, err := TrainDetector(ctx, m, ps, o); err == nil {
			t.Errorf("%+v accepted", o)
		}
	}
	if _, err := TrainDetector(ctx, m, nil, DetTrainOptions{Steps: 1, Batch: 1}); err == nil {
		t.Error("no page accepted")
	}
}

func TestSortLines(t *testing.T) {
	// A tall box with two small ones on its band (their centers a little
	// lower, their left edges further left), then a line below.
	b := []Box{
		{X0: 500, Y0: 0, X1: 600, Y1: 100},
		{X0: 100, Y0: 55, X1: 150, Y1: 59},
		{X0: 0, Y0: 50, X1: 50, Y1: 54},
		{X0: 0, Y0: 200, X1: 50, Y1: 220},
	}
	sortLines(b)
	var xs []float64
	for _, x := range b {
		xs = append(xs, x.X0)
	}
	if xs[0] != 0 || xs[1] != 100 || xs[2] != 500 || b[3].Y0 != 200 {
		t.Fatalf("order %v", b)
	}
}

func TestMixSourceWeights(t *testing.T) {
	s := NewMemorySource(pageLines(t), 1)
	for _, w := range [][]float64{{}, {1}, {0, 0}, {1, -1}} {
		if _, err := NewMixSource([]LineSource{s, s}, w, 1); err == nil {
			t.Errorf("weights %v accepted", w)
		}
	}
}

// TestTrainRejectsOtherHeights: lines cut for another model stop the run.
func TestTrainRejectsOtherHeights(t *testing.T) {
	v, _ := NewVocab(DefaultCharset)
	cfg := linerec.Base(len(v.Chars) + 1)
	cfg.Dim, cfg.Expand, cfg.SE, cfg.Blocks, cfg.CtxChannels, cfg.CtxDim = 16, 16, 4, 1, 4, 8
	m, _ := linerec.New(cfg, 1)
	l := ShardLine{Text: "ab", W: 40, Rows: 64, Ink: make([]float32, 64*40)}
	_, err := Train(context.Background(), m, v, NewMemorySource([]ShardLine{l}, 1), TrainOptions{Steps: 1, Batch: 1, LR: 1e-3})
	if err == nil || !strings.Contains(err.Error(), "pixels high") {
		t.Fatalf("err = %v", err)
	}
}

func TestExtendHead(t *testing.T) {
	old, _ := NewVocab("ab")
	nv, _ := NewVocab("bca")
	head := []float32{0, 0, 1, 1, 2, 2} // blank, a, b; dim 2
	bias := []float32{9, 10, 20}
	h, b := ExtendHead(head, bias, 2, old, nv, 1)
	// New order: blank, b, c, a.
	if h[2] != 2 || h[6] != 1 || b[1] != 20 || b[3] != 10 || b[0] != 9 || b[2] != 5 {
		t.Fatalf("head %v bias %v", h, b)
	}
}

func TestDetLossIgnore(t *testing.T) {
	m := linedet.New(linedet.Base(), 1)
	const H, W = 32, 32
	out := make([]float32, H*W*3)
	for i := range out {
		out[i] = 0.3
	}
	d := make([]float32, len(out))
	line := []Box{{X0: 4, Y0: 4, X1: 40, Y1: 12}}
	fig := []Box{{X0: 0, Y0: 30, X1: 64, Y1: 64}, {X0: 0, Y0: 0, X1: 64, Y1: 16}}
	detLoss(m, out, d, H, W, line, fig, 1)
	for y := 15; y < H; y++ {
		for x := range W {
			if d[(y*W+x)*3] != 0 {
				t.Fatalf("gradient %g at (%d, %d), in a figure", d[(y*W+x)*3], x, y)
			}
		}
	}
	// A labeled line inside an ignored region is learned all the same.
	if d[(4*W+10)*3] == 0 {
		t.Fatal("no gradient on a labeled line's core")
	}
	if d[(14*W+5)*3] == 0 {
		t.Fatal("no gradient between the regions")
	}
}
