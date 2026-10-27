package ocr

import (
	"image"
	"math"
	"testing"
)

func TestDeskew(t *testing.T) {
	// Lines of "words" on a page, then turned counterclockwise: they rise
	// to the right.
	page := image.NewGray(image.Rect(0, 0, 800, 1000))
	for i := range page.Pix {
		page.Pix[i] = 250
	}
	for y := 100; y < 900; y += 30 {
		for x := 80; x < 720; x += 70 {
			for dy := range 12 {
				for dx := range 55 {
					page.Pix[(y+dy)*page.Stride+x+dx] = 20
				}
			}
		}
	}
	if a := Skew(page); a != 0 {
		t.Fatalf("level page: skew %.2f", a)
	}
	slanted := Rotate(page, -2.5)
	straight, a := Deskew(slanted)
	if math.Abs(a-2.5) > 0.15 {
		t.Fatalf("skew %.2f, want 2.5", a)
	}
	if math.Abs(Skew(straight)) > 0.15 {
		t.Fatalf("still slanted: %.2f", Skew(straight))
	}
	if same, a := Deskew(page); same != page || a != 0 {
		t.Fatal("a level page is returned as is")
	}
}
