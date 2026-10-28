package ocr

import (
	"strings"
	"testing"

	"github.com/bornholm/indecis/internal/pdf"
)

// mg is a glyph of a formula: its text, font, origin, size and advance.
func mg(text, font string, x, base, size, adv float64) pdf.Glyph {
	return pdf.Glyph{Text: text, Known: true, Font: font, OX: x, OY: base, Size: size, Advance: adv, Dx: 1,
		X0: x, X1: x + adv, Y0: base - 0.75*size, Y1: base + 0.25*size}
}

// words draws text in a text font from x on, a glyph a letter.
func words(s string, x, base float64) []pdf.Glyph {
	var gs []pdf.Glyph
	for _, r := range s {
		if r == ' ' {
			gs = append(gs, pdf.Glyph{Text: " ", Known: true, Space: true, Dx: 1, OX: x, OY: base, Size: 10})
			x += 3
			continue
		}
		gs = append(gs, mg(string(r), "CMR10", x, base, 10, 5))
		x += 5
	}
	return gs
}

func formulas(gs []pdf.Glyph) []string {
	var out []string
	for _, g := range gs {
		if strings.HasPrefix(g.Text, "$") {
			out = append(out, g.Text)
		}
	}
	return out
}

func TestMathScripts(t *testing.T) {
	gs := words("the value ", 0, 100)
	gs = append(gs,
		mg("x", "CMMI10", 50, 100, 10, 5.7),
		mg("2", "CMR7", 56, 96, 7, 4),
		mg("+", "CMR10", 63, 100, 10, 7.8),
		mg("a", "CMMI10", 73, 100, 10, 5.3),
		mg("i", "CMMI7", 78.3, 101.5, 7, 2.9),
	)
	gs = append(gs, words(" is", 82, 100)...)
	out := mathGlyphs(gs, nil)
	got := formulas(out)
	if len(got) != 1 || got[0] != `$x^{2}+a_{i}$` {
		t.Fatalf("formulas %q", got)
	}
	var text strings.Builder
	for _, g := range out {
		if !strings.HasPrefix(g.Text, "$") {
			text.WriteString(g.Text)
		}
	}
	if text.String() != "the value  is" {
		t.Errorf("text left %q", text.String())
	}
}

func TestMathFractionRootAccent(t *testing.T) {
	gs := []pdf.Glyph{
		mg("y", "CMMI10", 0, 100, 10, 5),
		mg("=", "CMR10", 8, 100, 10, 7.8),
		mg("a", "CMMI10", 20, 93, 10, 5.3), // over the rule
		mg("b", "CMMI10", 20, 108, 10, 4.3), // under it
		mg("+", "CMR10", 30, 100, 10, 7.8),
		mg("√", "CMSY10", 40, 92, 10, 8.3), // hangs from the rule
		mg("x", "CMMI10", 48.3, 100, 10, 5.7),
		mg("+", "CMR10", 57, 100, 10, 7.8),
		{Text: "ˆ", Name: "circumflex", Known: true, Font: "CMR10", OX: 67.5, OY: 100, Size: 10, Advance: 5, Dx: 1, X0: 67.5, X1: 72.5, Y0: 92.5, Y1: 102.5},
		mg("u", "CMMI10", 67, 100, 10, 5.7),
	}
	rules := []pdf.Rule{
		{X0: 19, Y0: 97.3, X1: 26, Y1: 97.7}, // the fraction bar, on the math axis
		{X0: 48.3, Y0: 91.8, X1: 54, Y1: 92.2},
	}
	got := formulas(mathGlyphs(gs, rules))
	if len(got) != 1 || got[0] != `$y=\frac{a}{b}+\sqrt{x}+\hat{u}$` {
		t.Fatalf("formulas %q", got)
	}
}

func TestMathLimits(t *testing.T) {
	gs := []pdf.Glyph{
		mg("S", "CMMI10", 0, 100, 10, 6),
		mg("=", "CMR10", 8, 100, 10, 7.8),
		{Text: "∑", Name: "summationdisplay", Known: true, Font: "CMEX10", OX: 19, OY: 89, Size: 10, Advance: 14.4, Dx: 1, X0: 19, X1: 33.4, Y0: 88.6, Y1: 95},
		mg("i", "CMMI7", 21, 111, 7, 2.9),
		mg("=", "CMR7", 24, 111, 7, 6.2),
		mg("1", "CMR7", 30, 111, 7, 4),
		mg("n", "CMMI7", 24, 86, 7, 4.2),
		mg("x", "CMMI10", 36, 100, 10, 5.7),
		mg("i", "CMMI7", 41.7, 101.5, 7, 2.9),
	}
	got := formulas(mathGlyphs(gs, nil))
	if len(got) != 1 || got[0] != `$S=\sum_{i=1}^{n}x_{i}$` {
		t.Fatalf("formulas %q", got)
	}
}

func TestMathLongArrowAndOverlay(t *testing.T) {
	gs := []pdf.Glyph{
		mg("f", "CMMI10", 0, 100, 10, 4.9),
		mg(":", "CMR10", 7, 100, 10, 2.8),
		mg("A", "CMMI10", 12, 100, 10, 7.5),
		mg("−", "CMSY10", 22, 100, 10, 7.8),
		mg("→", "CMSY10", 28, 100, 10, 10), // \longrightarrow: a minus, then the arrow
		mg("B", "CMMI10", 41, 100, 10, 7.6),
		mg("∈", "CMSY10", 52, 100, 10, 6.7),
		{Text: "", Name: "negationslash", Known: true, Font: "CMSY10", OX: 52.5, OY: 100, Size: 10, Advance: 0, Dx: 1, X0: 52.5, X1: 58, Y0: 92.5, Y1: 102.5},
		mg("C", "CMMI10", 62, 100, 10, 7.2),
	}
	got := formulas(mathGlyphs(gs, nil))
	if len(got) != 1 || got[0] != `$f:A\longrightarrow B\not\in C$` {
		t.Fatalf("formulas %q", got)
	}
}

func TestMathBullets(t *testing.T) {
	// A list's bullet, set with a math font, is no formula.
	gs := append([]pdf.Glyph{mg("•", "CMSY10", 0, 100, 10, 5)}, words(" First item", 8, 100)...)
	if got := formulas(mathGlyphs(gs, nil)); len(got) != 0 {
		t.Fatalf("formulas %q", got)
	}
}

func TestJoinFormulaWords(t *testing.T) {
	for _, c := range [][2]string{
		{`$f(x) > 0$ for all $x \in X$. (3)`, `$f(x) > 0 \text{ for all } x \in X$. (3)`},
		{`$a = 1$, $b = 2$`, `$a = 1 , b = 2$`},
		{`Let $x$ be a point and $y$ another one.`, `Let $x$ be a point and $y$ another one.`},
		{`$a$ is a real number which is not $b$`, `$a$ is a real number which is not $b$`},
	} {
		if got := joinFormulaWords(c[0]); got != c[1] {
			t.Errorf("joinFormulaWords(%q) = %q, want %q", c[0], got, c[1])
		}
	}
}

func TestJoinFormulaRows(t *testing.T) {
	boxes := []Box{{100, 100, 300, 112}, {110, 116, 290, 128}, {50, 140, 500, 152}}
	lines := []Line{{Text: `$a = b + c$`}, {Text: `$= d$, (2)`}, {Text: `which ends the proof.`}}
	b, l := joinFormulaRows(boxes, lines)
	if len(l) != 2 || l[0].Text != `$\begin{aligned} a = b + c \\ = d \end{aligned}$ , (2)` || b[0] != (Box{100, 100, 300, 128}) {
		t.Fatalf("rows %v %v", l, b)
	}
}
