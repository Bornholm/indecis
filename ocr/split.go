package ocr

import (
	"slices"
	"strings"
)

// The gaps, between the centres of neighbouring characters and in line
// heights, at which a detected line is cut, as Indra OCR's: the detector
// merges the lines of two columns with a narrow gutter, or two cells.
const (
	splitGap    = 3.0 // cut at a wider gap
	gutterGap   = 1.6 // cut at a smaller gap when other lines' edges line up with it
	gutterEdges = 5   // how many other lines must end or start there (within half a line height)
	gutterWords = 1.4 // and the gap must be this many times the line's usual gap between words
)

// SplitLines cuts read lines at wide gaps between their characters. A
// piece keeps its characters, and its box is cut around them.
func SplitLines(boxes []Box, lines []Line) ([]Box, []Line) {
	var outB []Box
	var outL []Line
	for i, l := range lines {
		b := boxes[i]
		h := max(1, b.Y1-b.Y0)
		word := wordGap(l.Chars)
		var cuts []int // indexes of the characters that start a piece
		prev := -1
		for k, c := range l.Chars {
			if strings.TrimSpace(c.Text) == "" {
				continue
			}
			if prev >= 0 {
				a, x := center(l.Chars[prev]), center(c)
				gap := x - a
				if gap > splitGap*h || gap > gutterGap*h && gap > gutterWords*word && onGutter(boxes, i, a+0.35*h, x-0.35*h, h) {
					cuts = append(cuts, k)
				}
			}
			prev = k
		}
		if len(cuts) == 0 {
			outB, outL = append(outB, b), append(outL, l)
			continue
		}
		start := 0
		for _, end := range append(cuts, len(l.Chars)) {
			chars := l.Chars[start:end]
			start = end
			var sb strings.Builder
			x0, x1 := b.X1, b.X0
			for _, c := range chars {
				sb.WriteString(c.Text)
				if strings.TrimSpace(c.Text) != "" {
					x0, x1 = min(x0, center(c)), max(x1, center(c))
				}
			}
			text := strings.TrimSpace(sb.String())
			if text == "" {
				continue
			}
			outB = append(outB, Box{max(b.X0, x0-0.35*h), b.Y0, min(b.X1, x1+0.35*h), b.Y1})
			outL = append(outL, Line{Text: text, Chars: chars, Confidence: l.Confidence})
		}
	}
	return outB, outL
}

// wordGap is the median gap between the characters around a line's
// spaces (0 with fewer than two spaces): letter-spaced old prints have
// word gaps as wide as a narrow gutter.
func wordGap(chars []Char) float64 {
	var gaps []float64
	for k := 1; k+1 < len(chars); k++ {
		if strings.TrimSpace(chars[k].Text) == "" && strings.TrimSpace(chars[k-1].Text) != "" && strings.TrimSpace(chars[k+1].Text) != "" {
			gaps = append(gaps, center(chars[k+1])-center(chars[k-1]))
		}
	}
	if len(gaps) < 2 {
		return 0
	}
	slices.Sort(gaps)
	return gaps[len(gaps)/2]
}

func center(c Char) float64 { return (c.X0 + c.X1) / 2 }

// onGutter reports whether a gap of line i from gl to gr sits on a column
// gutter: enough other lines end at its left edge or start at its right.
func onGutter(boxes []Box, i int, gl, gr, h float64) bool {
	n := 0
	for j, o := range boxes {
		if j != i && (abs(o.X1-gl) < 0.5*h || abs(o.X0-gr) < 0.5*h) {
			n++
		}
	}
	return n >= gutterEdges
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
