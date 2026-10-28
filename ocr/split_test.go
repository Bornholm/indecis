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
	// A gap of more than 3 line heights is cut, a narrower one stays.
	b, l := SplitLines([]Box{{0, 0, 600, 20}}, []Line{readLine(chars("Total", 0), chars("12,50", 200))})
	if len(l) != 2 || l[0].Text != "Total" || l[1].Text != "12,50" || b[0].X1 > 60 || b[1].X0 < 190 {
		t.Errorf("wide gap: %+v %v", l, b)
	}
	_, l = SplitLines([]Box{{0, 0, 600, 20}}, []Line{readLine(chars("Total", 0), chars("12,50", 90))})
	if len(l) != 1 {
		t.Errorf("narrow gap cut: %+v", l)
	}
}
