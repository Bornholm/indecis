package ocr

import "testing"

// chars lays out text from x, one 10-pixel character per rune.
func chars(text string, x float64) []Char {
	var cs []Char
	for _, r := range text {
		cs = append(cs, Char{Text: string(r), X0: x, X1: x + 4, P: 1})
		x += 10
	}
	return cs
}

func readLine(parts ...[]Char) Line {
	var l Line
	for _, p := range parts {
		l.Chars = append(l.Chars, p...)
	}
	for _, c := range l.Chars {
		l.Text += c.Text
	}
	return l
}

func TestSplitLines(t *testing.T) {
	h := 20.0
	// A line merged across a gutter: its gap (from 290 to 340) is 2.5 line
	// heights, where five lines of each column end and start.
	boxes := []Box{{0, 0, 600, h}}
	lines := []Line{readLine(chars("un deux trois", 170), chars("quatre cinq", 340))}
	for i := range 5 {
		y := float64(30 * (i + 1))
		boxes = append(boxes, Box{100, y, 300, y + h}, Box{330, y, 600, y + h})
		lines = append(lines, readLine(chars("x", 100)), readLine(chars("y", 340)))
	}
	b, l := SplitLines(boxes, lines)
	if len(l) != len(lines)+1 || l[0].Text != "un deux trois" || l[1].Text != "quatre cinq" {
		t.Fatalf("lines %q", texts(l))
	}
	if b[0].X1 > 310 || b[1].X0 < 320 {
		t.Errorf("boxes %v %v", b[0], b[1])
	}
	// The same gap with no gutter around it stays.
	_, l = SplitLines(boxes[:1], lines[:1])
	if len(l) != 1 {
		t.Errorf("cut with no gutter: %q", texts(l))
	}
	// A gap of more than 3 line heights is always cut.
	_, l = SplitLines([]Box{{0, 0, 600, h}}, []Line{readLine(chars("Total", 0), chars("12,50", 200))})
	if len(l) != 2 {
		t.Errorf("wide gap not cut: %q", texts(l))
	}
}

func texts(l []Line) []string {
	var s []string
	for _, x := range l {
		s = append(s, x.Text)
	}
	return s
}
