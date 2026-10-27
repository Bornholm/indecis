package ocr

import (
	"image"
	"sort"
)

// Orient finds which way up a page is: scanned sideways or upside down,
// its lines read as nothing. It reads the page's widest lines as it is,
// and, unless they read surely, turned by a quarter, a half and three
// quarters: the way whose characters the recognizer is surest of wins.
// It returns the page turned upright, the clockwise quarter turns applied
// (0 to 3) and the lines found on it.
func Orient(d *Detector, r *Recognizer, page *image.Gray) (*image.Gray, int, []Box) {
	best, bestTurns, bestScore := page, 0, -1.0
	var bestBoxes []Box
	for turns := range 4 {
		g := page
		if turns > 0 {
			g = RotateQuarter(page, turns)
		}
		boxes := d.Detect(g)
		score, sure := readScore(r, g, boxes)
		if score > bestScore {
			best, bestTurns, bestScore, bestBoxes = g, turns, score, boxes
		}
		if turns == 0 && sure {
			break
		}
	}
	return best, bestTurns, bestBoxes
}

// orientLines is how many of a page's widest lines Orient reads.
const orientLines = 12

// readScore reads a page's widest lines and returns the sum of their
// characters' probabilities, and whether they read surely: at least 20
// characters, at 0.9 on average.
func readScore(r *Recognizer, page *image.Gray, boxes []Box) (float64, bool) {
	sel := append([]Box(nil), boxes...)
	sort.Slice(sel, func(i, j int) bool { return sel[i].X1-sel[i].X0 > sel[j].X1-sel[j].X0 })
	sel = sel[:min(len(sel), orientLines)]
	score, n := 0.0, 0
	for _, l := range r.ReadLines(page, sel) {
		for _, c := range l.Chars {
			score += float64(c.P)
			n++
		}
	}
	return score, n >= 20 && score >= 0.9*float64(n)
}

// RotateQuarter turns a page clockwise by turns quarter turns.
func RotateQuarter(page *image.Gray, turns int) *image.Gray {
	b := page.Bounds()
	w, h := b.Dx(), b.Dy()
	turns = ((turns % 4) + 4) % 4
	at := func(x, y int) uint8 { return page.Pix[page.PixOffset(b.Min.X+x, b.Min.Y+y)] }
	var out *image.Gray
	switch turns {
	case 0:
		out = image.NewGray(image.Rect(0, 0, w, h))
		for y := range h {
			copy(out.Pix[y*w:(y+1)*w], page.Pix[page.PixOffset(b.Min.X, b.Min.Y+y):])
		}
	case 1: // (x, y) -> (h-1-y, x)
		out = image.NewGray(image.Rect(0, 0, h, w))
		for y := range h {
			for x := range w {
				out.Pix[x*h+h-1-y] = at(x, y)
			}
		}
	case 2:
		out = image.NewGray(image.Rect(0, 0, w, h))
		for y := range h {
			for x := range w {
				out.Pix[(h-1-y)*w+w-1-x] = at(x, y)
			}
		}
	case 3: // (x, y) -> (y, w-1-x)
		out = image.NewGray(image.Rect(0, 0, h, w))
		for y := range h {
			for x := range w {
				out.Pix[(w-1-x)*h+y] = at(x, y)
			}
		}
	}
	return out
}
