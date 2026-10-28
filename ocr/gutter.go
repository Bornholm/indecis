package ocr

import (
	"image"
	"sort"
)

// The detector merges the lines of two columns set close (a gutter of a
// line height or less, as narrow as a justified line's widest spaces). On
// the page, the gutter is a blank band down many lines at once, where a
// space between words lines up with others for a few lines at most.
const (
	gutterMinWidth = 0.8 // narrowest gutter, in line heights
	gutterNear     = 8   // lines within this many line heights above or below...
	gutterAlong    = 6   // ...at least this many of which run across it blank
	gutterInk      = 200 // grey level below which a pixel is ink
)

// splitBoxGutters cuts the boxes that run across a column gutter.
func splitBoxGutters(page *image.Gray, boxes []Box) []Box {
	if len(boxes) < gutterAlong+1 {
		return boxes
	}
	hs := make([]float64, len(boxes))
	for i, b := range boxes {
		hs[i] = b.Y1 - b.Y0
	}
	sort.Float64s(hs)
	h := hs[len(hs)/2]
	if h < 4 {
		return boxes
	}
	// Each box's blank bands inside it, wide enough to be a gutter.
	type band struct{ a, b float64 }
	blanks := make([][]band, len(boxes))
	for i, b := range boxes {
		x0, x1 := int(b.X0), int(b.X1)
		y0, y1 := int(b.Y0+0.15*(b.Y1-b.Y0)), int(b.Y1-0.15*(b.Y1-b.Y0))
		r := image.Rect(x0, y0, x1, y1).Intersect(page.Rect)
		if r.Dx() < int(4*h) || r.Dy() < 1 {
			continue
		}
		ink := make([]bool, r.Dx())
		for y := r.Min.Y; y < r.Max.Y; y++ {
			row := page.Pix[(y-page.Rect.Min.Y)*page.Stride-page.Rect.Min.X:]
			for x := r.Min.X; x < r.Max.X; x++ {
				if row[x] < gutterInk {
					ink[x-r.Min.X] = true
				}
			}
		}
		first, last := -1, -1
		for k, v := range ink {
			if v {
				if first < 0 {
					first = k
				}
				last = k
			}
		}
		start := -1
		for k := first; k <= last && first >= 0; k++ {
			if !ink[k] {
				if start < 0 {
					start = k
				}
				continue
			}
			if start >= 0 && float64(k-start) >= gutterMinWidth*h {
				blanks[i] = append(blanks[i], band{float64(r.Min.X + start), float64(r.Min.X + k)})
			}
			start = -1
		}
	}
	var out []Box
	for i, b := range boxes {
		var cuts []band
		for _, g := range blanks[i] {
			c := (g.a + g.b) / 2
			n := 0
			for j, o := range boxes {
				if j == i || o.Y1 < b.Y0-gutterNear*h || o.Y0 > b.Y1+gutterNear*h {
					continue
				}
				// A line of either column ending or starting at the gutter.
				if abs(o.X1-g.a) < 0.5*h || abs(o.X0-g.b) < 0.5*h {
					n++
					continue
				}
				if o.X0 > c-h || o.X1 < c+h {
					continue
				}
				for _, og := range blanks[j] {
					if og.a <= c && og.b >= c {
						n++
						break
					}
				}
			}
			if n >= gutterAlong {
				cuts = append(cuts, g)
			}
		}
		x0 := b.X0
		for _, g := range cuts {
			out = append(out, Box{x0, b.Y0, g.a + 0.1*h, b.Y1})
			x0 = g.b - 0.1*h
		}
		out = append(out, Box{x0, b.Y0, b.X1, b.Y1})
	}
	return out
}
