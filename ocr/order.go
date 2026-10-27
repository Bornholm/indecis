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
// gutter, and the cut above the columns comes first.
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
