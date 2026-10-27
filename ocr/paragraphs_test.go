package ocr

import "testing"

func TestParagraphs(t *testing.T) {
	boxes := []Box{
		{0, 0, 300, 10}, {0, 14, 280, 24}, // a paragraph of two lines
		{0, 60, 200, 70},   // another, far below
		{400, 60, 600, 70}, // further along the same line
		{0, 200, 100, 210}, // a last paragraph
	}
	lines := []Line{{Text: "the first trav-"}, {Text: "elling line"}, {Text: "second"}, {Text: "third"}, {Text: "last"}}
	got := Paragraphs(boxes, lines)
	want := []string{"the first travelling line", "second third", "last"}
	if len(got) != len(want) {
		t.Fatalf("%q", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%q", got)
		}
	}
	if h := PageHTML([]string{"a < b"}); h != "<p>a &lt; b</p>\n" {
		t.Fatalf("%q", h)
	}
}
