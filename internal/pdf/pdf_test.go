package pdf

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"strings"
	"testing"
)

// build writes a PDF file of objects (numbered from 1, the first the
// catalog) with a correct xref table.
func build(objs ...string) []byte {
	var b bytes.Buffer
	b.WriteString("%PDF-1.4\n")
	offs := make([]int, len(objs))
	for i, o := range objs {
		offs[i] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, o)
	}
	x := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objs)+1)
	for _, o := range offs {
		fmt.Fprintf(&b, "%010d 00000 n \n", o)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objs)+1, x)
	return b.Bytes()
}

func stream(dict, data string) string {
	return fmt.Sprintf("<< %s /Length %d >>\nstream\n%s\nendstream", dict, len(data), data)
}

func deflated(data string) string {
	var b bytes.Buffer
	w := zlib.NewWriter(&b)
	w.Write([]byte(data))
	w.Close()
	return stream("/Filter /FlateDecode", b.String())
}

func page(content string, extra ...string) []byte {
	objs := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 600 800] >>",
		"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> >> /Contents 4 0 R >>",
		content,
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
	}
	if len(extra) > 0 {
		objs = append(objs, extra...)
	} else {
		objs = append(objs, objs[4])
	}
	return build(objs...)
}

func text(gs []Glyph) string {
	var s strings.Builder
	for _, g := range gs {
		s.WriteString(g.Text)
	}
	return s.String()
}

func glyphs(t *testing.T, data []byte) []Glyph {
	t.Helper()
	f, err := Open(data)
	if err != nil {
		t.Fatal(err)
	}
	ps := f.Pages()
	if len(ps) != 1 {
		t.Fatalf("%d pages, want 1", len(ps))
	}
	return ps[0].Glyphs()
}

func TestGlyphs(t *testing.T) {
	gs := glyphs(t, page(deflated("BT /F1 10 Tf 1 0 0 1 100 700 Tm (Hello) Tj [(W) 120 (orld)] TJ ET")))
	if got := text(gs); got != "HelloWorld" {
		t.Fatalf("text %q", got)
	}
	// Helvetica's H is 722/1000 wide: at 10 points, 7.22.
	h := gs[0]
	if d := h.X1 - h.X0; d < 7.2 || d > 7.25 || h.X0 != 100 {
		t.Errorf("H box %v..%v", h.X0, h.X1)
	}
	// Top-left origin: the baseline at y 700 is 100 points from the top.
	if h.Y1 < 100 || h.Y1 > 102.5 || h.Y0 > 93 {
		t.Errorf("H box y %v..%v", h.Y0, h.Y1)
	}
	// The TJ adjustment of 120/1000 em moves "orld" left of where W ends.
	w, o := gs[5], gs[6]
	if gap := o.X0 - w.X1; gap > -1.1 || gap < -1.3 {
		t.Errorf("W..o gap %v, want -1.2", gap)
	}
}

func TestDifferencesAndToUnicode(t *testing.T) {
	cmap := stream("", `/CIDInit /ProcSet findresource begin 12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
2 beginbfchar <01> <00660069> <02> <00E9> endbfchar
1 beginbfrange <03> <05> <0041> endbfrange
endcmap CMapName currentdict /CMap defineresource pop end end`)
	data := page(stream("", "BT /F2 10 Tf 0 0 Td (\x01\x02\x03\x05) Tj /F3 10 Tf (\x80\x81AB) Tj ET"),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 8 0 R >>",
		"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding << /Differences [128 /eacute /f_i 65 /uni00E7 /ccedilla] >> >>",
		cmap)
	if got := text(glyphs(t, data)); got != "fiéACéfiçç" {
		t.Fatalf("text %q", got)
	}
}

func TestType0Identity(t *testing.T) {
	cmap := stream("", "1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfrange <0010> <0012> <0061> endbfrange")
	data := page(stream("", "BT /F2 12 Tf 0 0 Td <001000110012> Tj ET"),
		"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 8 0 R /DescendantFonts [7 0 R] >>",
		"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /DW 500 /W [16 [600 700]] >>",
		cmap)
	gs := glyphs(t, data)
	if got := text(gs); got != "abc" {
		t.Fatalf("text %q", got)
	}
	if d := gs[1].X0 - gs[0].X0; d < 7.19 || d > 7.21 {
		t.Errorf("advance of the first glyph %v, want 7.2", d)
	}
	if d := gs[2].X0 - gs[1].X0; d < 8.39 || d > 8.41 {
		t.Errorf("advance of the second glyph %v, want 8.4", d)
	}
}

func TestActualTextAndInvisible(t *testing.T) {
	gs := glyphs(t, page(stream("", "BT /F1 10 Tf 3 Tr /Span << /ActualText (x) >> BDC (ab) Tj EMC ET")))
	if len(gs) != 1 || gs[0].Text != "x" || !gs[0].Invisible {
		t.Fatalf("glyphs %+v", gs)
	}
}

func TestRotatedPage(t *testing.T) {
	data := build(
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Rotate 90 /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
		stream("", "BT /F1 10 Tf 100 700 Td (A) Tj ET"),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	)
	f, err := Open(data)
	if err != nil {
		t.Fatal(err)
	}
	p := f.Pages()[0]
	if w, h := p.Size(); w != 800 || h != 600 {
		t.Fatalf("size %vx%v", w, h)
	}
	g := p.Glyphs()[0]
	// Turned clockwise, the baseline runs down the page.
	if g.Dx > 0.01 || g.Dy < 0.99 {
		t.Errorf("direction %v,%v", g.Dx, g.Dy)
	}
	// The page's top left corner goes top right: the glyph's ascent points
	// right of its baseline, 700 points from the left.
	if g.X0 < 697 || g.X0 > 699 || g.X1 < 706 || g.X1 > 708 || g.Y0 < 99.9 || g.Y0 > 100.1 {
		t.Errorf("box %v,%v %v,%v", g.X0, g.Y0, g.X1, g.Y1)
	}
}

func TestBrokenXref(t *testing.T) {
	data := page(stream("", "BT /F1 10 Tf (ok) Tj ET"))
	i := bytes.LastIndex(data, []byte("startxref"))
	data = append(data[:i], "startxref\n99999\n%%EOF\n"...)
	if got := text(glyphs(t, data)); got != "ok" {
		t.Fatalf("text %q", got)
	}
}

func TestFilters(t *testing.T) {
	if got := string(ascii85([]byte("87cURD]i,\"Ebo80~>"))); got != "Hello World!" {
		t.Errorf("ascii85 %q", got)
	}
	if got := string(runLength([]byte{2, 'a', 'b', 'c', 254, 'x', 128})); got != "abcxxx" {
		t.Errorf("run length %q", got)
	}
	// "-----A---B" in LZW, from the example of the PDF reference.
	if got := string(lzw([]byte{0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01}, true)); got != "-----A---B" {
		t.Errorf("lzw %q", got)
	}
}

// FuzzOpen checks that no file, however broken, makes the reader panic.
func FuzzOpen(f *testing.F) {
	f.Add(page(deflated("BT /F1 10 Tf 1 0 0 1 100 700 Tm (Hello) Tj [(W) 120 (orld)] TJ ET")))
	f.Add(page(stream("", "BT /F2 12 Tf 0 0 Td <001000110012> Tj ET"),
		"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 8 0 R /DescendantFonts [7 0 R] >>",
		"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /DW 500 /W [16 [600 700]] >>",
		stream("", "1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfrange <0010> <0012> <0061> endbfrange")))
	f.Fuzz(func(t *testing.T, data []byte) {
		file, err := Open(data)
		if err != nil {
			return
		}
		for _, p := range file.Pages() {
			p.Size()
			p.Glyphs()
		}
	})
}
