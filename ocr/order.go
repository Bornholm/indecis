package ocr

import (
	"sort"
)

// ReadingOrder returns the indices of boxes in reading order, by recursive
// XY-cut: a set of lines is cut in two along its widest blank band, across
// (between blocks, read top to bottom) or down (between columns, read left
// to right), until no band is left; then top to bottom and left to right.
// A column gutter must be wider than a gap between blocks to win (1.5
// against 0.5 line height): a heading across two columns blocks the
// gutter, and the cut above the columns comes first. A narrower gutter
// (old prints set columns a third of a line height apart) cuts only
// between columns of text: wide, of several lines each, side by side.
// Short cells facing each other row by row (a receipt's labels and
// amounts) are a table, read row by row: no cut between them.
func ReadingOrder(boxes []Box) []int {
	idx := make([]int, len(boxes))
	for i := range idx {
		idx[i] = i
	}
	if len(boxes) == 0 {
		return idx
	}
	hs := make([]float64, len(boxes))
	for i, b := range boxes {
		hs[i] = b.Y1 - b.Y0
	}
	sort.Float64s(hs)
	h := max(hs[len(hs)/2], 1)
	var out []int
	xyCut(boxes, idx, h, &out)
	return out
}

func xyCut(boxes []Box, idx []int, h float64, out *[]int) {
	if len(idx) <= 1 {
		*out = append(*out, idx...)
		return
	}
	ys, yGap := cuts(boxes, idx, func(b Box) (float64, float64) { return b.Y0, b.Y1 }, 0.5*h)
	xs, xGap := cuts(boxes, idx, func(b Box) (float64, float64) { return b.X0, b.X1 }, 1.5*h)
	if len(xs) == 0 {
		if nx, gap := cuts(boxes, idx, func(b Box) (float64, float64) { return b.X0, b.X1 }, 0.25*h); len(nx) > 0 && gap >= yGap && textColumns(boxes, idx, nx[0], h) {
			xs, xGap = nx, 1.5*gap
		}
	}
	if len(xs) > 0 && tableRows(boxes, idx, xs[0], h) {
		if cellsOnly(boxes, idx, h) {
			*out = append(*out, tableOrder(boxes, idx, h)...)
			return
		}
		xs = nil
	}
	switch {
	case len(xs) > 0 && xGap/1.5 >= yGap:
		for _, part := range split(boxes, idx, xs, func(b Box) float64 { return (b.X0 + b.X1) / 2 }) {
			xyCut(boxes, part, h, out)
		}
	case len(ys) > 0:
		for _, part := range split(boxes, idx, ys, func(b Box) float64 { return (b.Y0 + b.Y1) / 2 }) {
			xyCut(boxes, part, h, out)
		}
	default:
		part := make([]Box, len(idx))
		for i, j := range idx {
			part[i] = boxes[j]
		}
		sorted := append([]Box(nil), part...)
		sortLines(sorted)
		// Map the sorted boxes back to their indices (boxes may repeat).
		used := make([]bool, len(part))
		for _, b := range sorted {
			for i, p := range part {
				if !used[i] && p == b {
					used[i] = true
					*out = append(*out, idx[i])
					break
				}
			}
		}
	}
}

// textColumns reports whether the boxes left and right of x make two
// columns of text: at least four lines each, most of them wider than 8
// line heights, each side running down at least half the other's height.
func textColumns(boxes []Box, idx []int, x, h float64) bool {
	type side struct {
		n, wide int
		y0, y1  float64
	}
	var l, r side
	l.y0, r.y0 = 1e18, 1e18
	for _, j := range idx {
		b := boxes[j]
		s := &l
		if (b.X0+b.X1)/2 > x {
			s = &r
		}
		s.n++
		if b.X1-b.X0 >= 8*h {
			s.wide++
		}
		s.y0, s.y1 = min(s.y0, b.Y0), max(s.y1, b.Y1)
	}
	if l.n < 4 || r.n < 4 || 2*l.wide < l.n || 2*r.wide < r.n {
		return false
	}
	overlap := min(l.y1, r.y1) - max(l.y0, r.y0)
	return overlap >= 0.5*min(l.y1-l.y0, r.y1-r.y0)
}

// tableRows reports whether the boxes left and right of x face each other
// row by row (most boxes of each side have one on the other side at the
// same height) while one side is made of short cells (most under 8 line
// heights).
func tableRows(boxes []Box, idx []int, x, h float64) bool {
	var l, r []Box
	for _, j := range idx {
		if b := boxes[j]; (b.X0+b.X1)/2 > x {
			r = append(r, b)
		} else {
			l = append(l, b)
		}
	}
	short := func(s []Box) bool {
		n := 0
		for _, b := range s {
			if b.X1-b.X0 < 8*h {
				n++
			}
		}
		return 2*n > len(s)
	}
	facing := func(s, other []Box) bool {
		n := 0
		for _, a := range s {
			for _, b := range other {
				if min(a.Y1, b.Y1)-max(a.Y0, b.Y0) >= 0.5*min(a.Y1-a.Y0, b.Y1-b.Y0) {
					n++
					break
				}
			}
		}
		return 10*n >= 7*len(s)
	}
	return (short(l) || short(r)) && facing(l, r) && facing(r, l)
}

// cellsOnly reports whether the boxes are a table's cells alone: nine in
// ten under 8 line heights wide (no line of text running beside it).
func cellsOnly(boxes []Box, idx []int, h float64) bool {
	n := 0
	for _, j := range idx {
		if boxes[j].X1-boxes[j].X0 < 8*h {
			n++
		}
	}
	return 10*n >= 9*len(idx)
}

// tableOrder reads a table row by row, each cell whole: the lines of a
// cell wrapped on several lines come before the next cell. As Indra OCR
// does, a row of lines joins the row above when it is a continuation: its
// gap is under half the table's median gap between rows and under 0.6 line
// height, and it has fewer lines than the row above (it fills only the
// wrapped cells). In a row, lines stacked one over another make one cell.
func tableOrder(boxes []Box, idx []int, h float64) []int {
	sorted := append([]int(nil), idx...)
	sort.SliceStable(sorted, func(a, b int) bool {
		return boxes[sorted[a]].Y0+boxes[sorted[a]].Y1 < boxes[sorted[b]].Y0+boxes[sorted[b]].Y1
	})
	// Rows: lines whose centres are within half a line height of the row's.
	var rows [][]int
	var centre []float64
	for _, j := range sorted {
		c := (boxes[j].Y0 + boxes[j].Y1) / 2
		if n := len(rows); n > 0 && abs(c-centre[n-1]) < 0.5*h {
			rows[n-1] = append(rows[n-1], j)
			centre[n-1] += (c - centre[n-1]) / float64(len(rows[n-1]))
			continue
		}
		rows, centre = append(rows, []int{j}), append(centre, c)
	}
	top := func(r []int) float64 {
		y := boxes[r[0]].Y0
		for _, j := range r {
			y = min(y, boxes[j].Y0)
		}
		return y
	}
	bottom := func(r []int) float64 {
		y := boxes[r[0]].Y1
		for _, j := range r {
			y = max(y, boxes[j].Y1)
		}
		return y
	}
	if len(rows) >= 3 {
		gaps := make([]float64, len(rows)-1)
		for i := range gaps {
			gaps[i] = top(rows[i+1]) - bottom(rows[i])
		}
		med := append([]float64(nil), gaps...)
		sort.Float64s(med)
		m := med[len(med)/2]
		merged := [][]int{rows[0]}
		prev := rows[0]
		for i, r := range rows[1:] {
			if g := gaps[i]; g < 0.5*m && g < 0.6*h && len(r) < len(prev) {
				merged[len(merged)-1] = append(merged[len(merged)-1], r...)
			} else {
				merged = append(merged, r)
			}
			prev = r
		}
		rows = merged
	}
	var out []int
	for _, r := range rows {
		// Cells: lines overlapping horizontally, left to right; in a cell,
		// top to bottom.
		sort.SliceStable(r, func(a, b int) bool { return boxes[r[a]].X0 < boxes[r[b]].X0 })
		var cells [][]int
		end := 0.0
		for _, j := range r {
			b := boxes[j]
			if n := len(cells); n > 0 && b.X0 < end {
				cells[n-1] = append(cells[n-1], j)
				end = max(end, b.X1)
				continue
			}
			cells = append(cells, []int{j})
			end = b.X1
		}
		for _, c := range cells {
			sort.SliceStable(c, func(a, b int) bool { return boxes[c[a]].Y0 < boxes[c[b]].Y0 })
			out = append(out, c...)
		}
	}
	return out
}

// cuts returns the middle of the widest blank band of the projection of
// the boxes on one axis, if it is at least minGap wide, and its width. One
// cut at a time: cutting at every band at once would cut two columns into
// rows at their line gaps.
func cuts(boxes []Box, idx []int, span func(Box) (float64, float64), minGap float64) ([]float64, float64) {
	type iv struct{ a, b float64 }
	ivs := make([]iv, len(idx))
	for i, j := range idx {
		a, b := span(boxes[j])
		ivs[i] = iv{a, b}
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].a < ivs[j].a })
	var mids []float64
	widest := 0.0
	end := ivs[0].b
	for _, v := range ivs[1:] {
		if gap := v.a - end; gap >= minGap && gap > widest {
			mids = []float64{(v.a + end) / 2}
			widest = gap
		}
		end = max(end, v.b)
	}
	return mids, widest
}

// split groups idx by the bands cut at mids, in order, by each box's
// position along the axis.
func split(boxes []Box, idx []int, mids []float64, pos func(Box) float64) [][]int {
	parts := make([][]int, len(mids)+1)
	for _, j := range idx {
		p := sort.SearchFloat64s(mids, pos(boxes[j]))
		parts[p] = append(parts[p], j)
	}
	return parts
}
