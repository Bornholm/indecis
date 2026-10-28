package ocr

import (
	"image"
)

// Orient finds which way up a page is: scanned sideways or upside down,
// its lines read as nothing. It reads the page's lines as it is and,
// unless they read surely, turned by a quarter, a half and three quarters:
// the way whose characters the recognizer is surest of wins. It returns
// the page turned upright, the clockwise quarter turns applied (0 to 3),
// the lines found on it and, when it read them all, their readings (nil
// otherwise).
func Orient(d *Detector, r *Recognizer, page *image.Gray) (*image.Gray, int, []Box, []Line) {
	best, bestTurns, bestScore := page, 0, -1.0
	var bestBoxes []Box
	var bestLines []Line
	uprightSure := false
	for turns := range 4 {
		g := page
		if turns > 0 {
			g = RotateQuarter(page, turns)
		}
		boxes := d.Detect(g)
		score, sure, lines := readScore(r, g, boxes)
		if turns == 0 {
			uprightSure = sure
			best, bestScore, bestBoxes, bestLines = g, score, boxes, lines
			if sure {
				break
			}
			continue
		}
		// A turned page must read clearly better than as it is, or read
		// surely where it did not: on a page whose lines go unseen, a few
		// characters read anyhow would otherwise turn it.
		if score > bestScore*orientMargin || (sure && !uprightSure && score > bestScore) {
			best, bestTurns, bestScore, bestBoxes, bestLines = g, turns, score, boxes, lines
		}
	}
	return best, bestTurns, bestBoxes, bestLines
}

// orientLines is how many of a page's lines Orient reads at most (all of
// them, but on the densest pages: a few lines picked out, the widest or
// the most typical, may be a scan's streaks or minute labels, and the
// upright page lose); orientMargin, how much surer a turned reading must
// be than the upright one.
const (
	orientLines  = 400
	orientMargin = 1.5
)

// readScore reads up to orientLines of a page's lines and returns the sum
// of their characters' probabilities, whether they read surely (at least
// 20 characters, at 0.9 on average), and the readings when they are all
// the page's lines.
func readScore(r *Recognizer, page *image.Gray, boxes []Box) (float64, bool, []Line) {
	lines := r.ReadLines(page, boxes[:min(len(boxes), orientLines)])
	score, n := 0.0, 0
	for _, l := range lines {
		for _, c := range l.Chars {
			score += float64(c.P)
			n++
		}
	}
	if len(lines) < len(boxes) {
		lines = nil
	}
	return score, n >= 20 && score >= 0.9*float64(n), lines
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
