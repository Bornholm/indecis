package ocr

import (
	"testing"

	"github.com/bornholm/indecis/internal/pdf"
)

// word lays out text from x on the baseline y, 10-point glyphs 5 points
// wide.
func word(text string, x, y float64) []pdf.Glyph {
	var gs []pdf.Glyph
	for _, r := range text {
		gs = append(gs, pdf.Glyph{Text: string(r), X0: x, X1: x + 5, Y0: y - 8, Y1: y + 2, Dx: 1, Size: 10, Known: true})
		x += 5
	}
	return gs
}

func layerTexts(t *testing.T, gs []pdf.Glyph) []string {
	t.Helper()
	l := layerLines(gs, 600, 800)
	var out []string
	for _, line := range l.Lines {
		out = append(out, line.Text)
	}
	return out
}

func TestLayerLines(t *testing.T) {
	var gs []pdf.Glyph
	gs = append(gs, word("Deux", 100, 100)...)
	gs = append(gs, word("mots", 123, 100)...)    // a 3-point gap: a space
	gs = append(gs, word("colonne", 300, 100)...) // far right: another line
	gs = append(gs, word("suite", 100, 112)...)
	// A superscript, raised and smaller, goes on its line.
	gs = append(gs, pdf.Glyph{Text: "2", X0: 125, X1: 128, Y0: 103, Y1: 109, Dx: 1, Size: 6, Known: true})
	got := layerTexts(t, gs)
	want := []string{"Deux mots", "suite2", "colonne"} // the left column first
	if len(got) != len(want) {
		t.Fatalf("lines %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("line %d %q, want %q", i, got[i], want[i])
		}
	}
}

func TestLayerAccents(t *testing.T) {
	// TeX draws é as an acute accent over an e, before or after it; its box
	// is the font's, as tall as a letter's.
	gs := word("d", 100, 100)
	gs = append(gs, pdf.Glyph{Text: "´", X0: 106, X1: 109, Y0: 91, Y1: 101, Dx: 1, Size: 10, Known: true})
	gs = append(gs, word("ej", 105, 100)...)
	gs = append(gs, word("a", 115, 100)...)
	gs = append(gs, pdf.Glyph{Text: "`", X0: 115.5, X1: 118.5, Y0: 91, Y1: 101, Dx: 1, Size: 10, Known: true})
	got := layerTexts(t, gs)
	if len(got) != 1 || got[0] != "déjà" {
		t.Fatalf("lines %q", got)
	}
}

func TestLayerChecks(t *testing.T) {
	gs := word("scan", 100, 100)
	for i := range gs {
		gs[i].Invisible = true
	}
	if l := layerLines(gs, 600, 800); l.Trusted {
		t.Errorf("an invisible layer is trusted")
	}
	gs = word("cassé", 100, 100)
	gs[2].Known = false
	if l := layerLines(gs, 600, 800); l.Trusted {
		t.Errorf("a layer with unknown glyphs is trusted")
	}
}

func TestLayerTurned(t *testing.T) {
	// A page set sideways: its lines run up the page.
	var gs []pdf.Glyph
	for i, r := range "Table" {
		y := 500 - 5*float64(i)
		gs = append(gs, pdf.Glyph{Text: string(r), X0: 100, X1: 110, Y0: y - 5, Y1: y, Dy: -1, Size: 10, Known: true})
	}
	l := layerLines(gs, 600, 800)
	if l.Turns != 1 || l.Width != 800 || l.Height != 600 {
		t.Fatalf("turns %d, size %vx%v", l.Turns, l.Width, l.Height)
	}
	if len(l.Lines) != 1 || l.Lines[0].Text != "Table" {
		t.Fatalf("lines %+v", l.Lines)
	}
	b := l.Boxes[0]
	if b.X1-b.X0 < 2*(b.Y1-b.Y0) {
		t.Errorf("turned line box %+v is not wide", b)
	}
}

func TestLayerGutter(t *testing.T) {
	// Two columns 8 points apart (0.8 em): lines are joined across the
	// gutter, then cut again where every line is blank.
	var gs []pdf.Glyph
	for i := range 5 {
		y := 100 + 12*float64(i)
		gs = append(gs, word("gauche", 100, y)...)
		gs = append(gs, word("un", 133, y)...)
		gs = append(gs, word("droite", 153, y)...)
		gs = append(gs, word("deux", 186, y)...)
	}
	got := layerTexts(t, gs)
	if len(got) != 10 || got[0] != "gauche un" || got[5] != "droite deux" {
		t.Fatalf("lines %q", got)
	}
}

func TestLayerTrustOCR(t *testing.T) {
	gs := word("scan", 100, 100)
	for i := range gs {
		gs[i].Invisible = true
	}
	l := layerLines(gs, 600, 800)
	if l.Trusted || !l.OCRLayer || len(l.Lines) != 1 {
		t.Fatalf("trusted %v, OCR layer %v, %d lines", l.Trusted, l.OCRLayer, len(l.Lines))
	}
	l.TrustOCR()
	if !l.Trusted || l.Lines[0].Text != "scan" {
		t.Fatalf("trusted %v, lines %+v", l.Trusted, l.Lines)
	}
}
