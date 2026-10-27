package ocr

import (
	"html"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Paragraphs groups the lines of a page, given in reading order, into
// paragraphs: a line joins the previous one's paragraph when it goes on
// along the same line, or starts below it within 0.8 line height and
// overlaps it horizontally. A word cut
// by a hyphen at a line's end is joined again when the next line goes on
// in lower case.
func Paragraphs(boxes []Box, lines []Line) []string {
	var out []string
	var cur strings.Builder
	for i, l := range lines {
		text := strings.TrimSpace(l.Text)
		if i > 0 && cur.Len() > 0 {
			a, b := boxes[i-1], boxes[i]
			h := min(a.Y1-a.Y0, b.Y1-b.Y0)
			gap := b.Y0 - a.Y1
			overlap := min(a.X1, b.X1) - max(a.X0, b.X0)
			// A piece further along the same line (the detector cuts at
			// wide gaps) stays in the paragraph.
			sameLine := min(a.Y1, b.Y1)-max(a.Y0, b.Y0) > 0.5*h && b.X0 >= a.X1-0.5*h
			if !sameLine && (gap > 0.8*h || gap < -0.5*h || overlap <= 0) {
				out = append(out, cur.String())
				cur.Reset()
			}
		}
		if text == "" {
			continue
		}
		if cur.Len() > 0 {
			s := cur.String()
			first, _ := firstRune(text)
			before, _ := utf8.DecodeLastRuneInString(strings.TrimSuffix(s, "-"))
			if strings.HasSuffix(s, "-") && unicode.IsLetter(before) && unicode.IsLower(first) {
				cur.Reset()
				cur.WriteString(strings.TrimSuffix(s, "-"))
			} else {
				cur.WriteByte(' ')
			}
		}
		cur.WriteString(text)
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

func firstRune(s string) (rune, bool) {
	for _, r := range s {
		return r, true
	}
	return 0, false
}

// PageHTML is a page's paragraphs as HTML, one <p> each.
func PageHTML(paragraphs []string) string {
	var b strings.Builder
	for _, p := range paragraphs {
		b.WriteString("<p>")
		b.WriteString(html.EscapeString(p))
		b.WriteString("</p>\n")
	}
	return b.String()
}
