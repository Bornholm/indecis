package ocr

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bornholm/indecis/internal/linedet"
)

// TestDetectorLearnsPage: a detector learns the test page's five lines
// (all but one, in this short a run).
func TestDetectorLearnsPage(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model")
	}
	page, cases := loadPage(t)
	ps := PageSample{Image: filepath.Join(pageDir, "page.png")}
	for _, c := range cases {
		ps.Boxes = append(ps.Boxes, [4]float64{c.Box[0], c.Box[1], c.Box[2], c.Box[3]})
		ps.Texts = append(ps.Texts, c.Text)
	}
	m := linedet.New(linedet.Base(), 1)
	var log strings.Builder
	_, err := TrainDetector(context.Background(), m, []PageSample{ps}, DetTrainOptions{
		Steps: 500, Batch: 4, LR: 5e-3, Warmup: 20, Clip: 5, Crop: 256, MinSide: 900, MaxSide: 1100,
		LogEvery: 50, Log: &log,
	})
	t.Log("\n" + log.String())
	if err != nil {
		t.Fatal(err)
	}
	d := NewDetector(m)
	got := d.Detect(page)
	ref := make([]Box, len(ps.Boxes))
	for i, b := range ps.Boxes {
		ref[i] = Box{b[0], b[1], b[2], b[3]}
	}
	matched := 0
	for i, j := range Match(ref, got, 0.5) {
		if j >= 0 {
			matched++
			t.Logf("line %d: %+v ~ %+v (IoU %.2f)", i, ref[i], got[j], IoU(ref[i], got[j]))
		}
	}
	if matched < len(ref)-1 || len(got) > len(ref)+1 {
		t.Fatalf("%d of %d lines found, %d boxes: %+v", matched, len(ref), len(got), got)
	}
}
