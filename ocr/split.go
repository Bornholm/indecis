package ocr

import "strings"

// splitGap is the gap between the centres of neighbouring characters, in
// line heights, at which a read line is cut, as Indra OCR's: the detector
// merges two cells or a label and its amount. (Gutters between columns are
// cut at detection, from the page: splitBoxGutters.)
const splitGap = 3.0

// SplitLines cuts read lines at wide gaps between their characters. A
// piece keeps its characters, and its box is cut around them.
func SplitLines(boxes []Box, lines []Line) ([]Box, []Line) {
	var outB []Box
	var outL []Line
	for i, l := range lines {
		b := boxes[i]
		h := max(1, b.Y1-b.Y0)
		var cuts []int // indexes of the characters that start a piece
		prev := -1
		for k, c := range l.Chars {
			if strings.TrimSpace(c.Text) == "" {
				continue
			}
			if prev >= 0 && center(c)-center(l.Chars[prev]) > splitGap*h {
				cuts = append(cuts, k)
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

func center(c Char) float64 { return (c.X0 + c.X1) / 2 }

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
