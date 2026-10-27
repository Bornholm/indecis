package ocr

import (
	"image"
	"testing"
)

func TestRotateQuarter(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 3, 2))
	copy(g.Pix, []uint8{1, 2, 3, 4, 5, 6})
	want := map[int][]uint8{
		0: {1, 2, 3, 4, 5, 6},
		1: {4, 1, 5, 2, 6, 3}, // 2 wide, 3 high: the left column becomes the top row
		2: {6, 5, 4, 3, 2, 1},
		3: {3, 6, 2, 5, 1, 4},
	}
	for turns, w := range want {
		r := RotateQuarter(g, turns)
		if string(r.Pix) != string(w) {
			t.Fatalf("%d turns: %v, want %v", turns, r.Pix, w)
		}
		if back := RotateQuarter(r, 4-turns); string(back.Pix) != string(g.Pix) || back.Rect != g.Rect {
			t.Fatalf("%d turns and back: %v", turns, back.Pix)
		}
	}
}
