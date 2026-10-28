package ocr

import (
	"image"
	"testing"
)

func TestSplitBoxGutters(t *testing.T) {
	// Two columns of ten lines, 25 pixels apart (lines 20 high): words of
	// ink 40 wide with 10 between them. The detector merged the first
	// three lines across the gutter.
	page := image.NewGray(image.Rect(0, 0, 1000, 330))
	for i := range page.Pix {
		page.Pix[i] = 255
	}
	var boxes []Box
	for k := range 10 {
		y := 20 + 30*k
		for _, col := range [2][2]int{{50, 480}, {505, 935}} {
			for x := col[0]; x+40 <= col[1]; x += 50 {
				for yy := y + 4; yy < y+16; yy++ {
					for xx := x; xx < x+40; xx++ {
						page.Pix[yy*page.Stride+xx] = 0
					}
				}
			}
		}
		if k < 3 {
			boxes = append(boxes, Box{50, float64(y), 935, float64(y + 20)})
		} else {
			boxes = append(boxes, Box{50, float64(y), 470, float64(y + 20)}, Box{505, float64(y), 935, float64(y + 20)})
		}
	}
	got := splitBoxGutters(page, boxes)
	if len(got) != 20 {
		t.Fatalf("%d boxes, want 20: %v", len(got), got)
	}
	for _, b := range got {
		if b.X0 < 480 && b.X1 > 505 {
			t.Errorf("box %v still runs across the gutter", b)
		}
	}
	// Word gaps (10 pixels, half a line height) are never cut.
	if n := len(splitBoxGutters(page, boxes[:1])); n != 1 {
		t.Errorf("a lone line was cut in %d", n)
	}
}
