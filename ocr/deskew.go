package ocr

import (
	"image"
	"math"
	"slices"
)

// Deskew straightens a page scanned or photographed at a slant: the line
// detector finds level lines only, and a line tilted by its own height
// over its length is cut by its box. It returns the page turned by the
// opposite of its slant, larger so that no corner is lost, and the slant
// in degrees (positive when the lines rise to the right). A page within
// 0.3° of level is returned as is, with a slant of 0.
func Deskew(page *image.Gray) (*image.Gray, float64) {
	a := Skew(page)
	if math.Abs(a) < 0.3 {
		return page, 0
	}
	return Rotate(page, a), a
}

// Skew estimates the slant of a page's lines, in degrees, within ±8°: the
// angle at which the ink, summed along rows, is most concentrated (the
// projection profile method). Rows of text then fall on rows of the
// profile, gaps between them on empty ones.
func Skew(page *image.Gray) float64 {
	a, _ := skew(page)
	return a
}

// skew returns the slant and how much more concentrated the profile is at
// that angle than level.
func skew(page *image.Gray) (float64, float64) {
	// The ink of a page reduced to about 1,000 pixels on its longer side.
	w, h := page.Rect.Dx(), page.Rect.Dy()
	if w < 32 || h < 32 {
		return 0, 0
	}
	f := max(1, max(w, h)/1000)
	sw, sh := w/f, h/f
	small := make([]uint8, sw*sh)
	for y := range sh {
		for x := range sw {
			s := 0
			for dy := range f {
				row := page.Pix[(y*f+dy)*page.Stride+x*f:]
				for dx := range f {
					s += int(row[dx])
				}
			}
			small[y*sw+x] = uint8(s / (f * f))
		}
	}
	var hist [256]int
	for _, v := range small {
		hist[v]++
	}
	paper := percentile(&hist, len(small), 0.5)
	dark := percentile(&hist, len(small), 0.02)
	if paper-dark < 40 {
		return 0, 0
	}
	cut := uint8((paper + dark) / 2)
	// Ink pixels, relative to the center. Dense regions (photographs,
	// filled boxes) would weigh as much as all the text: only pixels with
	// paper next to them in their row count, the edges of strokes.
	var xs, ys []float32
	cx, cy := float32(sw)/2, float32(sh)/2
	for y := range sh {
		row := small[y*sw : (y+1)*sw]
		for x := 1; x < sw-1; x++ {
			if row[x] < cut && (row[x-1] >= cut || row[x+1] >= cut) {
				xs = append(xs, float32(x)-cx)
				ys = append(ys, float32(y)-cy)
			}
		}
	}
	if len(xs) < 200 {
		return 0, 0
	}
	diag := int(math.Hypot(float64(sw), float64(sh))) + 2
	best, gain := profileAngle(xs, ys, diag)
	// Columns set at different heights line up at a slant too, on a level
	// page: each half of it alone is level.
	var lx, ly, rx, ry []float32
	for i, x := range xs {
		if x < 0 {
			lx, ly = append(lx, x), append(ly, ys[i])
		} else {
			rx, ry = append(rx, x), append(ry, ys[i])
		}
	}
	if len(lx) >= 200 && len(rx) >= 200 {
		l, _ := profileAngle(lx, ly, diag)
		r, _ := profileAngle(rx, ry, diag)
		if math.Abs(l) < 0.3 && math.Abs(r) < 0.3 {
			return 0, 1
		}
	}
	return best, gain
}

// profileAngle returns the angle, within ±8°, at which the points xs, ys
// (relative to the page's center) are most concentrated on rows, and how
// much more than level. A line rising to the right by deg,
// y = y0 - x·tan(deg), falls on one row at deg.
func profileAngle(xs, ys []float32, diag int) (float64, float64) {
	prof := make([]float64, diag)
	score := func(deg float64) float64 {
		clear(prof)
		off := float32(diag / 2)
		t := float32(math.Tan(deg * math.Pi / 180))
		for i, x := range xs {
			// The row a pixel falls on once the page is turned by deg.
			r := int(ys[i] + t*x + off)
			if r >= 0 && r < diag {
				prof[r]++
			}
		}
		s := 0.0
		for _, v := range prof {
			s += v * v
		}
		return s
	}
	level := score(0)
	best, bestScore := 0.0, level
	for a := -8.0; a <= 8.0; a += 0.25 {
		if s := score(a); s > bestScore {
			best, bestScore = a, s
		}
	}
	for a := best - 0.2; a <= best+0.2; a += 0.05 {
		if s := score(a); s > bestScore {
			best, bestScore = a, s
		}
	}
	return best, bestScore / level
}

// Rotate turns a page by deg degrees clockwise (counterclockwise for
// negative deg), bilinearly, onto a page large enough to hold all of it;
// the corners are filled with the page's paper.
func Rotate(page *image.Gray, deg float64) *image.Gray {
	w, h := page.Rect.Dx(), page.Rect.Dy()
	sin, cos := math.Sincos(deg * math.Pi / 180)
	nw := int(math.Ceil(math.Abs(float64(w)*cos) + math.Abs(float64(h)*sin)))
	nh := int(math.Ceil(math.Abs(float64(w)*sin) + math.Abs(float64(h)*cos)))
	out := image.NewGray(image.Rect(0, 0, nw, nh))
	paper := borderMedian(page)
	cx, cy := float64(w)/2, float64(h)/2
	ncx, ncy := float64(nw)/2, float64(nh)/2
	for y := range nh {
		dy := float64(y) + 0.5 - ncy
		row := out.Pix[y*out.Stride : y*out.Stride+nw]
		for x := range nw {
			dx := float64(x) + 0.5 - ncx
			// The source of an output pixel: turned back by deg.
			sx := cos*dx + sin*dy + cx - 0.5
			sy := -sin*dx + cos*dy + cy - 0.5
			x0, y0 := int(math.Floor(sx)), int(math.Floor(sy))
			if x0 < 0 || y0 < 0 || x0+1 >= w || y0+1 >= h {
				row[x] = paper
				continue
			}
			fx, fy := sx-float64(x0), sy-float64(y0)
			p := page.Pix[y0*page.Stride+x0:]
			q := page.Pix[(y0+1)*page.Stride+x0:]
			top := float64(p[0])*(1-fx) + float64(p[1])*fx
			bot := float64(q[0])*(1-fx) + float64(q[1])*fx
			row[x] = uint8(top*(1-fy) + bot*fy + 0.5)
		}
	}
	return out
}

// borderMedian is the median gray of a page's outermost pixels.
func borderMedian(page *image.Gray) uint8 {
	w, h := page.Rect.Dx(), page.Rect.Dy()
	var v []uint8
	for x := range w {
		v = append(v, page.Pix[x], page.Pix[(h-1)*page.Stride+x])
	}
	for y := range h {
		v = append(v, page.Pix[y*page.Stride], page.Pix[y*page.Stride+w-1])
	}
	slices.Sort(v)
	return v[len(v)/2]
}
