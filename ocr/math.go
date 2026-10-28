package ocr

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/bornholm/indecis/internal/pdf"
)

// Formulas of a PDF's text layer, written as LaTeX: a born-digital page
// draws its formulas with math fonts (TeX's cmmi, cmsy and cmex, the AMS's
// msbm, Cambria Math), each symbol where it goes, and the fraction bars and
// the bars of roots as rules. A formula is the run of glyphs, in the order
// the page draws them, that holds glyphs of math fonts and goes on with
// glyphs that can be math (digits, operators, letters set small, function
// names); its layout is read back into LaTeX: fractions and roots around
// their rules, accents over their base, limits under and over big
// operators, scripts raised or lowered from the formula's baseline.

// mathFontPrefixes are the names of math fonts, upper case: TeX's, the
// AMS's, Euler, rsfs, Latin Modern's, txfonts and pxfonts, MathTime, and
// the Symbol font that mathptmx sets Greek with (StandardSymL).
var mathFontPrefixes = []string{
	"CMMI", "CMBSY", "CMSY", "CMEX", "MSAM", "MSBM", "EUFM", "EUFB", "EUSM", "EUSB", "EUEX", "EURM", "EURB",
	"RSFS", "ESINT", "LMMATH", "TXMI", "TXSY", "TXEX", "PXMI", "PXSY", "PXEX", "NTXMI", "NTXSY", "NTXEX", "NEWTXMI",
	"ZXLMI", "MTMI", "MTSY", "MTEX", "STMARY", "BBM", "BBOLD", "DSROM", "ESINT", "TEX-MATH", "RTXMI", "RPXMI",
	"RTXSY", "RPXSY", "STANDARDSYML",
}

// mathFont reports a font of math symbols, by its name.
func mathFont(name string) bool {
	u := strings.ToUpper(name)
	for _, p := range mathFontPrefixes {
		if strings.HasPrefix(u, p) {
			return true
		}
	}
	return strings.Contains(u, "MATH") // Cambria Math, STIX Two Math, Latin Modern Math
}

// extensionFont reports a font of big operators and delimiters, whose
// glyphs hang from their origin.
func extensionFont(name string) bool {
	u := strings.ToUpper(name)
	return strings.HasPrefix(u, "CMEX") || strings.HasPrefix(u, "EUEX") || strings.HasPrefix(u, "TEX-MATHX") || strings.HasPrefix(u, "MATHX") || strings.Contains(u, "EXTENSION") || strings.HasPrefix(u, "ESINT") ||
		(strings.HasPrefix(u, "TXEX") || strings.HasPrefix(u, "PXEX") || strings.HasPrefix(u, "NTXEX") || strings.HasPrefix(u, "MTEX"))
}

// fontVariant is the LaTeX alphabet that a font's letters stand for:
// blackboard bold, fraktur, script, calligraphic capitals.
func fontVariant(name string, r rune) string {
	u := strings.ToUpper(name)
	switch {
	case strings.HasPrefix(u, "MSBM") || strings.HasPrefix(u, "BBM") || strings.HasPrefix(u, "BBOLD") || strings.HasPrefix(u, "DSROM"):
		return `\mathbb`
	case strings.HasPrefix(u, "EUF"):
		return `\mathfrak`
	case strings.HasPrefix(u, "RSFS"):
		return `\mathscr`
	case (strings.HasPrefix(u, "EUS") || strings.HasPrefix(u, "CMSY") || strings.HasPrefix(u, "CMBSY") ||
		strings.HasPrefix(u, "LMMATHSYMBOLS")) && r >= 'A' && r <= 'Z':
		return `\mathcal`
	}
	return ""
}

// notSeeds are the symbols of math fonts that text uses too: bullets of
// lists, footnote marks.
const notSeeds = "•†‡§¶∗⋆·○□■♦♣♠♥✓©"

// opWords are the function names set upright in formulas.
var opWords = map[string]bool{
	"sin": true, "cos": true, "tan": true, "cot": true, "sec": true, "csc": true, "arcsin": true, "arccos": true,
	"arctan": true, "sinh": true, "cosh": true, "tanh": true, "coth": true, "log": true, "ln": true, "lg": true,
	"exp": true, "max": true, "min": true, "sup": true, "inf": true, "lim": true, "liminf": true, "limsup": true,
	"det": true, "dim": true, "ker": true, "deg": true, "gcd": true, "arg": true, "Pr": true, "hom": true,
	"mod": true, "tr": true, "Tr": true, "diag": true, "sgn": true, "sign": true, "rank": true, "span": true,
	"supp": true, "argmax": true, "argmin": true, "var": true, "Var": true, "cov": true, "Cov": true, "Re": true,
	"Im": true, "id": true, "dist": true, "vol": true, "div": true, "curl": true, "grad": true, "erf": true,
}

// limitWords take their limits under them, as big operators do.
var limitWords = map[string]bool{"lim": true, "liminf": true, "limsup": true, "max": true, "min": true,
	"sup": true, "inf": true, "argmax": true, "argmin": true, "det": true, "gcd": true, "Pr": true}

// stackable are the relations and arrows that text can be set over or
// under (\overset{\text{def}}{=}, \xrightarrow{f}).
var stackable = map[string]bool{"=": true, `\sim`: true, `\to`: true, `\leftarrow`: true, `\Rightarrow`: true,
	`\Leftarrow`: true, `\Leftrightarrow`: true, `\leftrightarrow`: true, `\longrightarrow`: true, `\approx`: true,
	`\equiv`: true, `\leq`: true, `\geq`: true, `\mapsto`: true, `\simeq`: true, `\cong`: true, `\hookrightarrow`: true}

const bigOps = "∑∏∐⋃⋂⨁⨂⨀⋁⋀⨄⨆∫∬∭∮"

// accents are the accents of formulas, by glyph name and by text.
var accentNames = map[string]string{
	"circumflex": `\hat`, "hat": `\hat`, "tilde": `\tilde`, "macron": `\bar`, "dotaccent": `\dot`,
	"dieresis": `\ddot`, "acute": `\acute`, "grave": `\grave`, "caron": `\check`, "breve": `\breve`,
	"ring": `\mathring`, "vector": `\vec`,
}

var accentTexts = map[string]string{
	"ˆ": `\hat`, "^": `\hat`, "̂": `\hat`, "˜": `\tilde`, "~": `\tilde`, "̃": `\tilde`, "¯": `\bar`,
	"̄": `\bar`, "˙": `\dot`, "̇": `\dot`, "¨": `\ddot`, "̈": `\ddot`, "´": `\acute`,
	"́": `\acute`, "`": `\grave`, "̀": `\grave`, "ˇ": `\check`, "̌": `\check`, "˘": `\breve`,
	"̆": `\breve`, "˚": `\mathring`, "⃗": `\vec`,
}

// accentName is the command of an accent glyph's name: hatwide is \hat.
func accentName(name string) string {
	if a, ok := accentNames[name]; ok {
		return a
	}
	for _, suf := range []string{"wide", "wider", "widest"} {
		if b, ok := strings.CutSuffix(name, suf); ok {
			return accentNames[b]
		}
	}
	return ""
}

// bigOperators are the big operators of extension fonts, which name their
// glyphs after the small ones (circleplusdisplay).
var bigOperators = map[string]string{"⊕": `\bigoplus`, "⊗": `\bigotimes`, "⊙": `\bigodot`, "∪": `\bigcup`,
	"∩": `\bigcap`, "∨": `\bigvee`, "∧": `\bigwedge`, "⊔": `\bigsqcup`, "⊎": `\biguplus`}

// delimiterPieces are the glyphs that big delimiters are built from: the
// top piece stands for the delimiter, the others for nothing.
var delimiterPieces = map[string]string{
	"parenlefttp": "(", "parenrighttp": ")", "bracketlefttp": "[", "bracketrighttp": "]",
	"bracelefttp": "{", "bracerighttp": "}", "parenleftbt": "", "parenrightbt": "", "parenleftex": "",
	"parenrightex": "", "bracketleftbt": "", "bracketrightbt": "", "bracketleftex": "", "bracketrightex": "",
	"braceleftbt": "", "bracerightbt": "", "braceleftmid": "", "bracerightmid": "", "braceex": "",
	"radicalbt": "√", "radicalvertex": "", "radicaltp": "", "arrowvertex": "", "arrowdblvertex": "",
	"vextendsingle": "|", "vextenddouble": "‖",
}

// mathGlyphs replaces the formulas among glyphs (in the order the page
// draws them) with one glyph each, whose text is the formula's LaTeX
// between dollars. rules are the page's rules.
func mathGlyphs(glyphs []pdf.Glyph, rules []pdf.Rule) []pdf.Glyph {
	seed := make([]bool, len(glyphs))
	found := false
	for i, g := range glyphs {
		if g.Space || g.Invisible || math.Abs(g.Dy) > 0.01 || g.Dx <= 0 || !mathFont(g.Font) {
			continue
		}
		t := glyphMath(g)
		if t == "" || strings.Contains(notSeeds, t) {
			continue
		}
		seed[i], found = true, true
	}
	if !found {
		return glyphs
	}
	var flat []pdf.Rule
	for _, r := range rules {
		if r.X1-r.X0 > 2*(r.Y1-r.Y0) {
			flat = append(flat, r)
		}
	}
	var out []pdf.Glyph
	done := 0 // glyphs before it are in out or a formula
	for i := 0; i < len(glyphs); i++ {
		if !seed[i] {
			continue
		}
		run := growRun(glyphs, seed, flat, i, done)
		if run == nil {
			continue
		}
		tex := formulaTeX(glyphs, run, flat)
		if tex == "" {
			continue
		}
		first := run[0]
		out = append(out, glyphs[done:first]...)
		in := make(map[int]bool, len(run))
		for _, k := range run {
			in[k] = true
		}
		last := run[len(run)-1]
		// Glyphs between the formula's that it left out (a space) go after it.
		var rest []pdf.Glyph
		for k := first; k <= last; k++ {
			if !in[k] {
				rest = append(rest, glyphs[k])
			}
		}
		out = append(out, formulaGlyph(glyphs, run, tex))
		out = append(out, rest...)
		done = last + 1
		i = last
	}
	return append(out, glyphs[done:]...)
}

// glyphMath is a glyph's text in a formula: a piece of a big delimiter
// stands for the delimiter, or for nothing.
func glyphMath(g pdf.Glyph) string {
	if t, ok := delimiterPieces[g.Name]; ok {
		return t
	}
	switch g.Name {
	case "negationslash":
		return "\u0338"
	case "arrowhookleft":
		return hookLeft
	case "arrowhookright":
		return hookRight
	case "mapstochar":
		return mapsTo
	}
	if accentName(g.Name) != "" && (g.Text == "" || accentTexts[g.Text] != "" || g.Text == "^" || g.Text == "~") {
		return accentMark
	}
	if !g.Known || privateUse(g.Text) || badText(g.Text) {
		return ""
	}
	return g.Text
}

// accentMark marks an accent glyph's text.
const accentMark = "\x00accent"

type runState struct {
	x0, x1, top, bot float64
	size             float64 // the size of the formula's text
	base             float64 // its first baseline
	right            float64 // the right edge of the last glyph
	opens            []delimSpan
}

// A delimSpan is a big opening delimiter of a formula, open still: the
// rows of a matrix or of cases are within its height.
type delimSpan struct {
	text         string
	x0, top, bot float64
}

func opener(t string) bool { return strings.Contains("([{⟨|‖", t) && t != "" }

func closer(t string) bool { return strings.Contains(")]}⟩|‖", t) && t != "" }

// growRun is the formula around glyph i, a seed: glyphs from done on that
// can be math, next to each other. nil when it holds no formula.
func growRun(gs []pdf.Glyph, seed []bool, rules []pdf.Rule, i, done int) []int {
	g := gs[i]
	top, bot := glyphSpan(g)
	st := runState{x0: g.OX, x1: glyphRight(g), top: top, bot: bot, size: g.Size, base: g.OY, right: glyphRight(g)}
	if extensionFont(g.Font) || glyphMath(g) == "√" {
		// A big symbol hangs from its origin: the formula's baseline is the
		// next glyph's.
		st.base = math.NaN()
	}
	run := []int{i}
	// Back: digits, operators and function names just before the seed.
	prepend := func(from, to int) {
		for j := to; j >= from; j-- {
			run = append([]int{j}, run...)
			st.add(gs[j])
		}
	}
	for k := i - 1; k >= done; k-- {
		p := gs[k]
		if p.Space || p.Invisible || math.Abs(p.Dy) > 0.01 || p.Dx <= 0 {
			break
		}
		t := glyphMath(p)
		w, n := wordBefore(gs, k, done)
		first := gs[run[0]]
		if limitWords[w] && first.OY > p.OY+0.3*p.Size && first.OX < glyphRight(p)+0.3*p.Size &&
			glyphRight(first) > gs[k-n+1].OX-0.3*p.Size {
			prepend(k-n+1, k) // an operator over its limits (\lim_{n\to\infty})
			k -= n - 1
			continue
		}
		// An accent, or a big operator over its limits, is drawn over the
		// formula's start.
		over := t == accentMark || accentTexts[t] != "" || extensionFont(p.Font)
		if !math.IsNaN(st.base) && !extensionFont(p.Font) && math.Abs(p.OY-st.base) > 0.6*max(p.Size, st.size) ||
			!over && glyphRight(p) > first.OX+0.1*st.size || first.OX-glyphRight(p) > 0.4*st.size {
			break
		}
		gap := first.OX - glyphRight(p)
		if n > 0 && isLatin(w) && (gap < 0.12*st.size || p.Size < 0.85*st.size && gap < 0.4*st.size ||
			operator(glyphMath(first)) && len([]rune(w)) <= 4 && !stopWords[strings.ToLower(w)]) {
			// Letters set close to the formula (a bold vector, an operator
			// name: \mathrm{PSL}_{2}), or before an operator.
			prepend(k-n+1, k)
			k -= n - 1
			continue
		}
		if n > 0 && opWords[w] {
			prepend(k-n+1, k)
			k -= n - 1
			continue
		}
		if !(seed[k] || mathish(t)) {
			break
		}
		prepend(k, k)
	}
	st.right = glyphRight(g)
	// On: glyphs that can be math, near the formula.
	for k := i + 1; k < len(gs); k++ {
		p := gs[k]
		if p.Space {
			continue
		}
		if p.Invisible || math.Abs(p.Dy) > 0.01 || p.Dx <= 0 || !st.near(p, rules) {
			break
		}
		if seed[k] || mathish(glyphMath(p)) {
			run = append(run, k)
			st.add(p)
			continue
		}
		t := glyphMath(p)
		if t == "" {
			if p.Known && p.Text == "" {
				// A piece of a delimiter that stands for nothing: the
				// delimiter is that much taller.
				if n := len(st.opens); n > 0 && math.Abs(p.OX-st.opens[n-1].x0) < 0.3*st.size {
					top, bot := glyphSpan(p)
					d := &st.opens[n-1]
					d.top, d.bot = min(d.top, top), max(d.bot, bot)
					st.top, st.bot = min(st.top, top), max(st.bot, bot)
				}
				continue
			}
			break
		}
		if !isLetters(t) {
			break
		}
		// Letters of a text font: set small (a script), a function name, a
		// single letter set close to the formula (\mathrm{d}x), a short
		// word after an operator (= \mathrm{U}(1)).
		w, n := wordAfter(gs, k)
		small := p.Size < 0.85*st.size
		glued := p.OX-st.right < 0.12*st.size
		next := k + n
		gluedNext := n == 1 && next < len(gs) && (seed[next] && gs[next].OX-glyphRight(gs[next-1]) < 0.12*st.size ||
			mathish(glyphMath(gs[next])) && !strings.Contains(".,;:", gs[next].Text) && gs[next].OX-glyphRight(gs[next-1]) < 0.2*st.size)
		afterOp := operator(glyphMath(gs[run[len(run)-1]])) && len([]rune(w)) <= 4 && !stopWords[strings.ToLower(w)]
		if !(small || opWords[w] || n == 1 && (glued || gluedNext) || afterOp) {
			break
		}
		for j := k; j < k+n; j++ {
			run = append(run, j)
			st.add(gs[j])
		}
		k += n - 1
	}
	run = trimRun(gs, run)
	for _, k := range run {
		if seed[k] {
			return run
		}
	}
	return nil
}

// glyphSpan is how far a glyph goes up and down the page. A big symbol
// of an extension font hangs from its origin, as tall as its size says
// (TeX's integraldisplay is 2.2 ems): font metrics do not tell.
func glyphSpan(g pdf.Glyph) (top, bot float64) {
	if !extensionFont(g.Font) {
		return g.Y0, g.Y1
	}
	h := 1.0
	integral := strings.HasPrefix(g.Name, "integral") || strings.HasPrefix(g.Name, "contintegral")
	switch {
	case strings.HasSuffix(g.Name, "display") && integral:
		h = 2.2
	case strings.HasSuffix(g.Name, "display"):
		h = 1.4
	case strings.HasSuffix(g.Name, "text") && integral:
		h = 1.1
	case strings.HasSuffix(g.Name, "Bigg"):
		h = 3.0
	case strings.HasSuffix(g.Name, "bigg"):
		h = 2.4
	case strings.HasSuffix(g.Name, "Big"):
		h = 1.8
	case strings.HasSuffix(g.Name, "big"):
		h = 1.2
	}
	return g.OY - 0.05*g.Size, g.OY + h*g.Size
}

func glyphRight(g pdf.Glyph) float64 {
	if g.Advance > 0 {
		return g.OX + g.Advance
	}
	return g.X1
}

func (st *runState) add(g pdf.Glyph) {
	st.x0, st.x1 = min(st.x0, g.OX), max(st.x1, glyphRight(g))
	top, bot := glyphSpan(g)
	st.top, st.bot = min(st.top, top), max(st.bot, bot)
	st.right = glyphRight(g)
	if t := glyphMath(g); extensionFont(g.Font) && (opener(t) || closer(t)) {
		n := len(st.opens)
		if closer(t) && n > 0 && (!opener(t) || st.opens[n-1].text == t) {
			st.opens = st.opens[:n-1]
		} else if opener(t) {
			st.opens = append(st.opens, delimSpan{t, g.OX, top, bot})
		}
	}
	if t := glyphMath(g); extensionFont(g.Font) || t == "√" || t == accentMark || accentTexts[t] != "" {
		return
	}
	// The baseline is the first glyph's of the formula's size: limits and
	// scripts drawn before their base do not set it.
	if math.IsNaN(st.base) && g.Size >= 0.9*st.size || g.Size > 1.1*st.size {
		st.base = g.OY
	}
	st.size = max(st.size, g.Size)
}

// near reports whether g goes on with the formula: to its right, or
// stacked on it (a fraction's denominator, a limit, an accent). A glyph of
// full size off the formula's baseline needs a rule next to it (a
// fraction bar, a root's bar): without, it is the next line's.
func (st *runState) near(g pdf.Glyph, rules []pdf.Rule) bool {
	s := st.size
	if top, bot := glyphSpan(g); top > st.bot+0.6*s || bot < st.top-0.6*s {
		return false
	}
	ruled := false
	for _, r := range rules {
		y := (r.Y0 + r.Y1) / 2
		if r.Y1-r.Y0 < 0.2*s && r.X0 < glyphRight(g) && r.X1 > g.OX && r.X0 < st.x1+1.6*s && r.X1 > st.x0-0.3*s &&
			y > g.OY-1.2*g.Size && y < g.OY+0.6*g.Size {
			ruled = true
			break
		}
	}
	if !ruled && (g.OX-st.x1 > 1.6*s || glyphRight(g) < st.x0-0.3*s) {
		return false
	}
	for _, d := range st.opens {
		if c := (g.Y0 + g.Y1) / 2; c > d.top && c < d.bot && g.OX > d.x0 {
			return true // a row of a matrix, of cases
		}
	}
	shift := g.OY - st.base
	if !ruled && !math.IsNaN(shift) && math.Abs(shift) > 0.5*s {
		// A formula that starts with a fraction has its numerator's
		// baseline: a glyph after the fraction is across its rule.
		lo, hi := min(g.OY, st.base), max(g.OY, st.base)
		for _, r := range rules {
			y := (r.Y0 + r.Y1) / 2
			if r.Y1-r.Y0 < 0.2*s && y > lo && y < hi && r.X0 < st.x1 && r.X1 > st.x0 && g.OX > r.X1-0.1*s {
				ruled = true
				break
			}
		}
	}
	return ruled || math.IsNaN(shift) || g.Size < 0.9*s || g.Size > 1.1*s || extensionFont(g.Font) ||
		math.Abs(shift) <= 0.5*s || glyphMath(g) == "√"
}

// mathish reports text that can be part of a formula set in a text font:
// digits, operators, punctuation, Greek letters.
func mathish(t string) bool {
	if t == "" || t == accentMark || t == hookLeft || t == hookRight || t == mapsTo {
		return t != ""
	}
	for _, r := range t {
		switch {
		case unicode.IsDigit(r), unicode.Is(unicode.Greek, r), unicode.Is(unicode.Sm, r):
		case strings.ContainsRune("()[]{}|/!',.:;*·×÷±∓′″−-+=<>◦†‡∗⋆#\u0338", r):
		default:
			return false
		}
	}
	return true
}

// operator reports a binary operator or a relation, which a formula goes
// on after.
func operator(t string) bool {
	for _, r := range t {
		if !(unicode.Is(unicode.Sm, r) || strings.ContainsRune("+-−=<>∈∉⊂⊆→←⇒↦×·", r)) {
			return false
		}
	}
	return t != ""
}

// stopWords are the short words of text that can follow a formula ending
// with an operator.
var stopWords = map[string]bool{"a": true, "an": true, "and": true, "or": true, "the": true, "for": true,
	"if": true, "iff": true, "of": true, "is": true, "are": true, "be": true, "in": true, "on": true, "at": true,
	"to": true, "by": true, "as": true, "so": true, "we": true, "it": true, "its": true, "all": true, "with": true,
	"then": true, "when": true, "that": true, "this": true, "has": true, "some": true, "each": true, "also": true,
	"from": true, "into": true, "not": true, "no": true, "but": true, "thus": true, "let": true, "le": true,
	"la": true, "les": true, "et": true, "ou": true, "de": true, "des": true, "du": true, "un": true, "une": true}

func isLatin(t string) bool {
	for _, r := range t {
		if !unicode.Is(unicode.Latin, r) {
			return false
		}
	}
	return t != ""
}

func isLetters(t string) bool {
	for _, r := range t {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return t != ""
}

// wordAfter is the word of letters from glyph k: same font and baseline,
// no gap; and how many glyphs it takes.
func wordAfter(gs []pdf.Glyph, k int) (string, int) {
	var sb strings.Builder
	n := 0
	for j := k; j < len(gs); j++ {
		g := gs[j]
		if g.Space || !isLetters(g.Text) || g.Font != gs[k].Font || math.Abs(g.OY-gs[k].OY) > 0.05*g.Size ||
			j > k && g.OX-glyphRight(gs[j-1]) > 0.1*g.Size {
			break
		}
		sb.WriteString(g.Text)
		n++
	}
	return sb.String(), n
}

// wordBefore is the word of letters that ends at glyph k.
func wordBefore(gs []pdf.Glyph, k, done int) (string, int) {
	j := k
	for j > done {
		p, g := gs[j-1], gs[j]
		if p.Space || !isLetters(p.Text) || p.Font != g.Font || math.Abs(p.OY-g.OY) > 0.05*g.Size ||
			g.OX-glyphRight(p) > 0.1*g.Size {
			break
		}
		j--
	}
	if !isLetters(gs[k].Text) {
		return "", 0
	}
	w, n := wordAfter(gs, j)
	if j+n-1 != k {
		return "", 0
	}
	return w, n
}

// trimRun leaves the punctuation that ends a sentence and unmatched
// brackets at the ends of a formula to the text.
func trimRun(gs []pdf.Glyph, run []int) []int {
	for len(run) > 0 {
		t := gs[run[len(run)-1]].Text
		open, close := count(gs, run, "(")+count(gs, run, "["), count(gs, run, ")")+count(gs, run, "]")
		if strings.Contains(".,;:([", t) || (t == ")" || t == "]") && open < close {
			run = run[:len(run)-1]
			continue
		}
		break
	}
	for len(run) > 0 {
		t := gs[run[0]].Text
		open, close := count(gs, run, "(")+count(gs, run, "["), count(gs, run, ")")+count(gs, run, "]")
		if (t == "(" || t == "[") && open > close {
			run = run[1:]
			continue
		}
		break
	}
	return run
}

func count(gs []pdf.Glyph, run []int, t string) int {
	n := 0
	for _, k := range run {
		if gs[k].Text == t {
			n++
		}
	}
	return n
}

// formulaGlyph is the glyph that stands for a formula.
func formulaGlyph(gs []pdf.Glyph, run []int, tex string) pdf.Glyph {
	f := gs[run[0]]
	f.Text, f.Known, f.Space, f.Name = "$"+tex+"$", true, false, ""
	size := 0.0
	for _, k := range run {
		g := gs[k]
		top, bot := glyphSpan(g)
		f.X0, f.Y0, f.X1, f.Y1 = min(f.X0, g.X0), min(f.Y0, top), max(f.X1, g.X1), max(f.Y1, bot)
		if !extensionFont(g.Font) {
			size = max(size, g.Size)
		}
	}
	if size > 0 {
		f.Size = size
	}
	return f
}

// An mnode is a part of a formula: a symbol, a word, or parts already put
// together (a fraction, a root, a symbol with its accent or limits).
type mnode struct {
	tex      string
	x0, x1   float64 // horizontal extent
	top, bot float64 // vertical extent
	base     float64 // baseline
	size     float64
	big      bool   // a big operator or delimiter, a root sign: never a script
	limits   bool   // takes limits under and over it in display
	accent   string // a lone accent: its command
	built    bool   // a fraction or the like: a script only when well off the baseline
}

// formulaTeX writes the formula of the glyphs of run in LaTeX.
func formulaTeX(gs []pdf.Glyph, run []int, rules []pdf.Rule) string {
	var ns []*mnode
	size := 0.0
	for k := 0; k < len(run); k++ {
		g := gs[run[k]]
		t := glyphMath(g)
		if t == "" {
			if _, piece := delimiterPieces[g.Name]; piece && len(ns) > 0 {
				// A piece of a tall delimiter: the delimiter goes down to it.
				for j := len(ns) - 1; j >= 0; j-- {
					if d := ns[j]; d.big && math.Abs(d.x0-g.OX) < 0.3*g.Size {
						top, bot := glyphSpan(g)
						d.top, d.bot = min(d.top, top), max(d.bot, bot)
						break
					}
				}
			}
			continue
		}
		n := &mnode{x0: g.OX, x1: glyphRight(g), base: g.OY, size: g.Size}
		if !extensionFont(g.Font) {
			size = max(size, g.Size)
		}
		switch {
		case t == accentMark:
			n.accent = accentName(g.Name)
		case t != "^" && t != "~":
			n.accent = accentTexts[t]
		}
		if !mathFont(g.Font) && isLatin(t) {
			// A word set upright: a function name or \mathrm.
			_, wn := wordAfter(gs, run[k])
			m := 1
			for m < wn && k+m < len(run) && run[k+m] == run[k]+m {
				m++
			}
			w, _ := wordAfter(gs[:run[k]+m], run[k])
			n.x1 = glyphRight(gs[run[k]+m-1])
			n.tex = wordTeX(w)
			n.limits = limitWords[w]
			k += m - 1
		} else if t == hookLeft || t == hookRight || t == mapsTo {
			n.tex = t
		} else if n.accent == "" {
			n.tex = symbolTeX(t, g.Font)
			n.big = extensionFont(g.Font) || t == "√"
			if op, ok := bigOperators[t]; ok && extensionFont(g.Font) {
				n.tex = op
			}
			n.limits = strings.Contains(bigOps, t) && !strings.HasPrefix(t, "∫") && t != "∮" || bigOperators[t] != "" && n.big ||
				stackable[n.tex]
		}
		switch {
		case extensionFont(g.Font):
			n.top, n.bot = glyphSpan(g)
		case t == "√":
			n.top, n.bot = g.OY-0.1*g.Size, g.OY+0.9*g.Size
		default:
			n.top, n.bot = g.OY-0.7*g.Size, g.OY+0.2*g.Size
		}
		ns = append(ns, n)
	}
	if len(ns) == 0 || size == 0 {
		return ""
	}
	ns = mergeRepeats(mergeOpWords(ns))
	// Rules within the formula, the shortest first: an inner fraction is
	// put together before the one around it.
	x0, x1, top, bot := ns[0].x0, ns[0].x1, ns[0].top, ns[0].bot
	for _, n := range ns {
		x0, x1, top, bot = min(x0, n.x0), max(x1, n.x1), min(top, n.top), max(bot, n.bot)
	}
	var rs []pdf.Rule
	for _, r := range rules {
		if r.Y1-r.Y0 < 0.15*size && r.X0 >= x0-0.3*size && r.X1 <= x1+0.3*size &&
			(r.Y0+r.Y1)/2 > top-0.2*size && (r.Y0+r.Y1)/2 < bot+0.2*size {
			rs = append(rs, r)
		}
	}
	sort.Slice(rs, func(i, j int) bool { return rs[i].X1-rs[i].X0 < rs[j].X1-rs[j].X0 })
	for _, r := range rs {
		ns = applyRule(ns, r, size)
	}
	return strings.TrimSpace(arrange(ns, size))
}

// opPairs are the function names set as two words.
var opPairs = map[string]string{`\lim \inf`: `\liminf`, `\lim \sup`: `\limsup`, `\arg \min`: `\operatorname*{argmin}`,
	`\arg \max`: `\operatorname*{argmax}`}

// mergeOpWords joins function names set as two words (lim inf, arg min).
func mergeOpWords(ns []*mnode) []*mnode {
	var out []*mnode
	for i := 0; i < len(ns); i++ {
		n := ns[i]
		if i+1 < len(ns) {
			m := ns[i+1]
			if t, ok := opPairs[n.tex+" "+m.tex]; ok && m.x0-n.x1 < 0.4*n.size && math.Abs(m.base-n.base) < 0.1*n.size {
				out = append(out, &mnode{tex: t, x0: n.x0, x1: m.x1, top: min(n.top, m.top), bot: max(n.bot, m.bot),
					base: n.base, size: n.size, limits: true})
				i++
				continue
			}
		}
		out = append(out, n)
	}
	return out
}

// mergeRepeats drops glyphs drawn again over themselves (bold made by
// drawing twice) and joins the pieces of a tall delimiter drawn one under
// the other.
func mergeRepeats(ns []*mnode) []*mnode {
	var out []*mnode
	for _, n := range ns {
		merged := false
		for _, m := range out {
			if m.tex != n.tex || m.accent != n.accent || math.Abs(m.x0-n.x0) > 0.1*n.size {
				continue
			}
			if math.Abs(m.base-n.base) < 0.1*n.size {
				merged = true // drawn twice
			} else if m.big && n.big && n.top <= m.bot+0.3*n.size && n.bot >= m.top-0.3*n.size {
				m.top, m.bot = min(m.top, n.top), max(m.bot, n.bot)
				merged = true // a piece of the same delimiter
			}
			if merged {
				break
			}
		}
		if !merged {
			out = append(out, n)
		}
	}
	return out
}

// applyRule puts together the parts around a rule: a root (its sign at the
// rule's left end), a fraction (parts over and under it), a line over or
// under parts.
func applyRule(ns []*mnode, r pdf.Rule, size float64) []*mnode {
	y := (r.Y0 + r.Y1) / 2
	within := func(n *mnode) bool {
		c := (n.x0 + n.x1) / 2
		return c > r.X0-0.1*size && c < r.X1+0.1*size
	}
	// A root: the sign ends where the rule starts.
	for ri, rad := range ns {
		if rad.tex != `\surd` || math.Abs(rad.x1-r.X0) > 0.3*rad.size || math.Abs(y-rad.top) > 0.3*rad.size {
			continue
		}
		var in, index []*mnode
		var rest []*mnode
		for i, n := range ns {
			switch {
			case i == ri:
			case within(n) && n.top >= y-0.3*size && n.top < rad.bot:
				in = append(in, n)
			case n.x1 <= rad.x0+0.6*rad.size && n.x1 > rad.x0-0.5*rad.size && n.bot < (rad.top+rad.bot)/2 && n.size < 0.9*rad.size:
				index = append(index, n)
			default:
				rest = append(rest, n)
			}
		}
		tex := `\sqrt{` + arrange(in, size) + `}`
		if len(index) > 0 {
			tex = `\sqrt[` + arrange(index, size) + `]{` + arrange(in, size) + `}`
		}
		m := &mnode{tex: tex, x0: rad.x0, x1: max(r.X1, rad.x1), top: r.Y0, bot: rad.bot, base: rad.base, size: rad.size, built: true}
		for _, n := range append(in, index...) {
			m.top, m.bot = min(m.top, n.top), max(m.bot, n.bot)
			m.base = max(m.base, n.base)
			m.size = max(m.size, n.size)
		}
		if len(in) > 0 {
			m.base = baseOf(in, size)
		}
		return append(rest, m)
	}
	// Parts over and under the rule, near it, and the parts that touch
	// them in its span.
	// The parts nearest the rule, then those that touch them and are no
	// larger (a script, a limit), never a part off to the side.
	dOver, dUnder := math.Inf(1), math.Inf(1)
	for _, n := range ns {
		if within(n) && n.bot <= y+0.15*n.size {
			dOver = min(dOver, y-n.bot)
		}
		if within(n) && n.top >= y-0.15*n.size {
			dUnder = min(dUnder, n.top-y)
		}
	}
	over := growParts(ns, within, func(n *mnode) bool {
		return n.bot <= y+0.15*n.size && y-n.bot < 0.8*n.size && y-n.bot < dOver+0.3*n.size
	}, func(n, m *mnode) bool {
		return n.bot <= y+0.15*n.size && n.bot > m.top-0.35*m.size && n.size <= 1.15*m.size
	})
	under := growParts(ns, within, func(n *mnode) bool {
		return n.top >= y-0.15*n.size && n.top-y < 0.8*n.size && n.top-y < dUnder+0.3*n.size
	}, func(n, m *mnode) bool {
		return n.top >= y-0.15*n.size && n.top < m.bot+0.35*m.size && n.size <= 1.15*m.size
	})
	var tex string
	var parts []*mnode
	switch {
	case len(over) > 0 && len(under) > 0:
		tex = `\frac{` + arrange(over, size) + `}{` + arrange(under, size) + `}`
		parts = append(over, under...)
	case len(under) > 0 && under[0].top-y < 0.5*size:
		tex = `\overline{` + arrange(under, size) + `}`
		parts = under
	case len(over) > 0 && y-over[0].bot < 0.3*size:
		tex = `\underline{` + arrange(over, size) + `}`
		parts = over
	default:
		return ns
	}
	m := &mnode{tex: tex, x0: r.X0, x1: r.X1, top: r.Y0, bot: r.Y1, size: 0, built: true}
	for _, n := range parts {
		m.x0, m.x1 = min(m.x0, n.x0), max(m.x1, n.x1)
		m.top, m.bot = min(m.top, n.top), max(m.bot, n.bot)
		m.size = max(m.size, n.size)
	}
	if len(over) > 0 && len(under) > 0 {
		m.base = y + 0.25*size // the fraction bar is on the math axis
	} else {
		m.base = baseOf(parts, size)
	}
	in := map[*mnode]bool{}
	for _, n := range parts {
		in[n] = true
	}
	var rest []*mnode
	for _, n := range ns {
		if !in[n] {
			rest = append(rest, n)
		}
	}
	return append(rest, m)
}

// growParts is the parts in span that start passes, and the parts in span that
// touch one of them.
func growParts(ns []*mnode, span func(*mnode) bool, start func(*mnode) bool, touch func(n, m *mnode) bool) []*mnode {
	var out []*mnode
	in := map[*mnode]bool{}
	for _, n := range ns {
		if span(n) && start(n) {
			out = append(out, n)
			in[n] = true
		}
	}
	for changed := len(out) > 0; changed; {
		changed = false
		for _, n := range ns {
			if in[n] || !span(n) {
				continue
			}
			for _, m := range out {
				if touch(n, m) {
					out = append(out, n)
					in[n], changed = true, true
					break
				}
			}
		}
	}
	return out
}

// baseOf is the main baseline of parts: the baseline most of the parts of
// the largest size sit on.
func baseOf(ns []*mnode, size float64) float64 {
	top := 0.0
	for _, n := range ns {
		if !n.big && n.accent == "" {
			top = max(top, n.size)
		}
	}
	best, bestN := 0.0, -1
	for _, n := range ns {
		if n.big || n.accent != "" || n.size < 0.9*top {
			continue
		}
		c := 0
		for _, m := range ns {
			if !m.big && m.accent == "" && m.size >= 0.9*top && math.Abs(m.base-n.base) < 0.15*top {
				c++
			}
		}
		if c > bestN || c == bestN && n.x0 < best {
			best, bestN = n.base, c
		}
	}
	if bestN < 0 {
		return ns[0].base
	}
	return best
}

type atom struct {
	tex      string
	big      bool
	sub, sup []*mnode
}

// arrange writes parts on a line: accents put on their base, limits on
// their operator, then from left to right, scripts after their base.
func arrange(ns []*mnode, size float64) string {
	if len(ns) == 0 {
		return ""
	}
	ns = applyArrows(applyOverlays(ns))
	ns = applyAccents(ns)
	ns = applyLimits(ns, false)
	ns = applyLimits(ns, true)
	ns = applyMatrices(ns, size)
	sort.SliceStable(ns, func(i, j int) bool { return ns[i].x0 < ns[j].x0 })
	s := 0.0
	for _, n := range ns {
		if !n.big && n.accent == "" {
			s = max(s, n.size)
		}
	}
	if s == 0 {
		s = size
	}
	main := baseOf(ns, size)
	var out []string
	var cur *atom
	flush := func() {
		if cur == nil {
			return
		}
		t := cur.tex
		if (len(cur.sub) > 0 || len(cur.sup) > 0) && strings.ContainsAny(t, "_^") {
			t = "{" + t + "}" // a base with scripts already: a double script is no LaTeX
		}
		sub, sup := `_{`+arrange(cur.sub, size)+`}`, `^{`+arrange(cur.sup, size)+`}`
		switch {
		case cur.big && len(cur.sub) > 0 && len(cur.sup) > 0:
			t += sub + sup
		case len(cur.sub) > 0 && len(cur.sup) > 0 && left(cur.sub) >= right(cur.sup):
			t += sup + `{}` + sub // staggered indices: R^{a}{}_{bcd}
		case len(cur.sub) > 0 && len(cur.sup) > 0 && left(cur.sup) >= right(cur.sub):
			t += sub + `{}` + sup
		default:
			if len(cur.sub) > 0 {
				t += sub
			}
			if len(cur.sup) > 0 {
				t += sup
			}
		}
		out = append(out, t)
		cur = nil
	}
	for _, n := range ns {
		shift := n.base - main
		lift := 0.1 * s
		if n.size < 0.8*s {
			lift = 0.05 * s
		}
		script := !n.big && (n.built && math.Abs(shift) > 0.25*s || !n.built && n.size < 0.9*s && math.Abs(shift) > lift ||
			!n.built && n.size <= 0.75*size && math.Abs(shift) > 0.3*s) // scripts of scripts keep their size
		prime := n.tex == `\prime` || n.tex == `\prime\prime`
		if prime && cur != nil && !strings.HasSuffix(cur.tex, `\prime`) {
			script, shift = true, -1 // a prime is a superscript
		}
		if n.accent != "" {
			continue // an accent with no base
		}
		if cur != nil && !n.big {
			// A script of a script: of the group it follows, whose glyphs
			// are larger.
			var into *[]*mnode
			best := math.Inf(1)
			for _, grp := range []*[]*mnode{&cur.sub, &cur.sup} {
				if len(*grp) == 0 || n.size >= 0.9*(*grp)[0].size {
					continue
				}
				last := (*grp)[len(*grp)-1]
				d := n.x0 - right(*grp)
				if v := math.Abs(d) + 0.5*math.Abs((n.top+n.bot)/2-(last.top+last.bot)/2); d > -0.3*n.size && v < best {
					into, best = grp, v
				}
			}
			if into != nil {
				*into = append(*into, n)
				continue
			}
		}
		if !script {
			flush()
			cur = &atom{tex: n.tex, big: n.big || n.limits}
			continue
		}
		if cur == nil {
			cur = &atom{tex: "{}"}
		}
		if shift < 0 {
			cur.sup = append(cur.sup, n)
		} else {
			cur.sub = append(cur.sub, n)
		}
	}
	flush()
	return joinTeX(dots(out))
}

// dots writes three dots in a row as an ellipsis.
func dots(ts []string) []string {
	var out []string
	for i := 0; i < len(ts); i++ {
		if i+2 < len(ts) && ts[i] == ts[i+1] && ts[i] == ts[i+2] && (ts[i] == "." || ts[i] == `\cdot`) {
			if ts[i] == "." {
				out = append(out, `\ldots`)
			} else {
				out = append(out, `\cdots`)
			}
			i += 2
			continue
		}
		out = append(out, ts[i])
	}
	return out
}

func left(ns []*mnode) float64 {
	x := math.Inf(1)
	for _, n := range ns {
		x = min(x, n.x0)
	}
	return x
}

func right(ns []*mnode) float64 {
	x := math.Inf(-1)
	for _, n := range ns {
		x = max(x, n.x1)
	}
	return x
}

// overlays are the symbols drawn over each other that make one.
var overlays = map[[2]string]string{{`\sim`, `=`}: `\cong`, {`\sim`, `-`}: `\simeq`, {`=`, `.`}: `\doteq`,
	{`/`, `\in`}: `\notin`, {`/`, `=`}: `\neq`}

// applyOverlays joins symbols drawn over each other: a negation slash and
// its relation, ∼ over = (\cong).
func applyOverlays(ns []*mnode) []*mnode {
	used := map[*mnode]bool{}
	var out []*mnode
	for i, a := range ns {
		for _, b := range ns[i+1:] {
			if used[a] || used[b] || a.built || b.built || a.accent != "" || b.accent != "" {
				continue
			}
			ov := min(a.x1, b.x1) - max(a.x0, b.x0)
			if ov < 0.6*min(a.x1-a.x0, b.x1-b.x0) || math.Abs(a.base-b.base) > 0.6*max(a.size, b.size) {
				continue
			}
			tex := overlays[[2]string{a.tex, b.tex}]
			if tex == "" {
				tex = overlays[[2]string{b.tex, a.tex}]
			}
			if tex == "" && math.Abs(a.base-b.base) > 0.3*max(a.size, b.size) {
				continue
			}
			switch {
			case tex != "":
			case a.tex == `\not` || a.tex == "/":
				tex = joinTeX([]string{`\not`, b.tex})
			case b.tex == `\not` || b.tex == "/":
				tex = joinTeX([]string{`\not`, a.tex})
			default:
				continue
			}
			used[a], used[b] = true, true
			out = append(out, &mnode{tex: tex, x0: min(a.x0, b.x0), x1: max(a.x1, b.x1), top: min(a.top, b.top),
				bot: max(a.bot, b.bot), base: max(a.base, b.base), size: max(a.size, b.size)})
		}
	}
	if len(out) == 0 {
		return ns
	}
	for _, n := range ns {
		if !used[n] {
			out = append(out, n)
		}
	}
	return out
}

// arrowPieces are the glyphs that TeX builds long arrows from, end to end:
// \longrightarrow is a minus sign and an arrow, \Longrightarrow an equals
// sign and a double arrow, \hookrightarrow a hook and an arrow.
var arrowPieces = map[string]bool{"-": true, "=": true, `\to`: true, `\leftarrow`: true, `\Rightarrow`: true,
	`\Leftarrow`: true, hookLeft: true, hookRight: true, mapsTo: true, `\mapsto`: true}

const (
	hookLeft  = "\x00arrowhookleft"
	hookRight = "\x00arrowhookright"
	mapsTo    = "\x00mapstochar"
)

// applyArrows joins the pieces of long arrows.
func applyArrows(ns []*mnode) []*mnode {
	sort.SliceStable(ns, func(i, j int) bool { return ns[i].x0 < ns[j].x0 })
	var out []*mnode
	for i := 0; i < len(ns); i++ {
		n := ns[i]
		j := i
		for arrowPieces[n.tex] && !n.built && j+1 < len(ns) && arrowPieces[ns[j+1].tex] && !ns[j+1].built &&
			ns[j+1].x0 < ns[j].x1+0.03*n.size && math.Abs(ns[j+1].base-n.base) < 0.1*n.size {
			j++
		}
		if tex := longArrow(ns[i : j+1]); tex != "" {
			last := ns[j]
			out = append(out, &mnode{tex: tex, x0: n.x0, x1: last.x1, top: min(n.top, last.top), bot: max(n.bot, last.bot),
				base: last.base, size: max(n.size, last.size), limits: true})
			i = j
			continue
		}
		if n.tex != hookLeft && n.tex != hookRight && n.tex != mapsTo {
			out = append(out, n)
		}
	}
	return out
}

// longArrow is the arrow that pieces end to end make, "" if none.
func longArrow(ps []*mnode) string {
	if len(ps) < 2 {
		return ""
	}
	first, last := ps[0].tex, ps[len(ps)-1].tex
	if first == `\mapsto` {
		first = mapsTo
	}
	right := last == `\to` || last == `\Rightarrow`
	left := first == `\leftarrow` || first == `\Leftarrow`
	double, minus := false, 0
	for _, p := range ps {
		double = double || p.tex == "=" || p.tex == `\Rightarrow` || p.tex == `\Leftarrow`
		if p.tex == "-" {
			minus++
		}
	}
	switch {
	case first == hookLeft && right:
		return `\hookrightarrow`
	case last == hookRight && left:
		return `\hookleftarrow`
	case first == mapsTo && right && len(ps) == 2:
		return `\mapsto`
	case first == mapsTo && right:
		return `\longmapsto`
	case left && right && double:
		return `\Longleftrightarrow`
	case left && right:
		return `\longleftrightarrow`
	case right && double:
		return `\Longrightarrow`
	case right && minus > 1:
		return `\to` // stretched under or over text: \xrightarrow
	case right && minus == 1:
		return `\longrightarrow`
	case left && double:
		return `\Longleftarrow`
	case left && minus > 0:
		return `\longleftarrow`
	}
	return ""
}

// applyMatrices writes the rows of parts stacked within a tall delimiter
// (and its partner, but for cases) as a matrix.
func applyMatrices(ns []*mnode, size float64) []*mnode {
	sort.SliceStable(ns, func(i, j int) bool { return ns[i].x0 < ns[j].x0 })
	for i := 0; i < len(ns); i++ {
		l := ns[i]
		if !l.big || !opener(delimText(l.tex)) || l.bot-l.top < 1.5*size {
			continue
		}
		var r *mnode
		for _, n := range ns[i+1:] {
			if n.big && closer(delimText(n.tex)) && min(n.bot, l.bot)-max(n.top, l.top) > 0.6*min(n.bot-n.top, l.bot-l.top) {
				r = n
				break
			}
		}
		if r == nil && l.tex != `\{` {
			continue
		}
		var cells, rest []*mnode
		for _, n := range ns {
			c, m := (n.x0+n.x1)/2, (n.top+n.bot)/2
			if n != l && n != r && c > l.x1 && (r == nil || c < r.x0) && m > l.top && m < l.bot {
				cells = append(cells, n)
			} else {
				rest = append(rest, n)
			}
		}
		tex := matrixTeX(cells, size)
		if tex == "" {
			continue
		}
		m := &mnode{tex: tex, x0: l.x1, x1: l.x1, top: l.top, bot: l.bot, base: (l.top+l.bot)/2 + 0.25*size, size: size, big: true}
		for _, n := range cells {
			m.x1 = max(m.x1, n.x1)
		}
		return applyMatrices(append(rest, m), size)
	}
	return ns
}

// delimText is a delimiter's text, from its LaTeX.
func delimText(tex string) string {
	switch tex {
	case `\{`:
		return "{"
	case `\}`:
		return "}"
	case `\langle`:
		return "⟨"
	case `\rangle`:
		return "⟩"
	case `\|`:
		return "‖"
	}
	return tex
}

// matrixTeX writes parts as the rows (two or more) of a matrix, its
// columns where wide gaps cut the rows; "" when they make one row.
func matrixTeX(cells []*mnode, size float64) string {
	top := 0.0
	for _, n := range cells {
		if !n.big {
			top = max(top, n.size)
		}
	}
	var bases []float64
	for _, n := range cells {
		if !n.big && n.size >= 0.85*top {
			bases = append(bases, n.base)
		}
	}
	sort.Float64s(bases)
	var rows []float64 // the baselines of the rows
	for _, b := range bases {
		if len(rows) == 0 || b-rows[len(rows)-1] > 0.5*size {
			rows = append(rows, b)
		}
	}
	if len(rows) < 2 {
		return ""
	}
	parts := make([][]*mnode, len(rows))
	for _, n := range cells {
		k := 0
		for j, b := range rows {
			if math.Abs(n.base-b) < math.Abs(n.base-rows[k]) {
				k = j
			}
		}
		parts[k] = append(parts[k], n)
	}
	var sb strings.Builder
	sb.WriteString(`\begin{matrix} `)
	for i, row := range parts {
		if i > 0 {
			sb.WriteString(` \\ `)
		}
		sort.SliceStable(row, func(a, b int) bool { return row[a].x0 < row[b].x0 })
		start := 0
		for k := 1; k <= len(row); k++ {
			if k == len(row) || row[k].x0-right(row[:k]) > 0.8*size {
				if start > 0 {
					sb.WriteString(` & `)
				}
				sb.WriteString(arrange(row[start:k], size))
				start = k
			}
		}
	}
	sb.WriteString(` \end{matrix}`)
	return sb.String()
}

// applyAccents puts each accent on the part it is drawn over.
func applyAccents(ns []*mnode) []*mnode {
	used := map[*mnode]bool{}
	var out []*mnode
	for _, a := range ns {
		if a.accent == "" {
			continue
		}
		var base *mnode
		best := 0.0
		for _, b := range ns {
			if b.accent != "" || used[b] {
				continue
			}
			ov := min(a.x1, b.x1) - max(a.x0, b.x0)
			if ov > 0.3*min(a.x1-a.x0, b.x1-b.x0) && ov > best && b.base >= a.base-0.05*b.size {
				base, best = b, ov
			}
		}
		if base == nil {
			continue
		}
		if a.x1-a.x0 > 1.3*(base.x1-base.x0) {
			// A wide accent over a base and its scripts: \widetilde{P_{1}}.
			parts := []*mnode{base}
			for _, b := range ns {
				c := (b.x0 + b.x1) / 2
				if b != base && b.accent == "" && !used[b] && c > a.x0 && c < a.x1 && b.base >= a.base-0.05*b.size {
					parts = append(parts, b)
				}
			}
			if len(parts) > 1 {
				m := &mnode{tex: a.accent + `{` + arrange(parts, base.size) + `}`, x0: a.x0, x1: a.x1, top: a.top,
					bot: base.bot, base: base.base, size: base.size, built: base.built}
				for _, b := range parts {
					used[b] = true
					m.x0, m.x1, m.bot = min(m.x0, b.x0), max(m.x1, b.x1), max(m.bot, b.bot)
				}
				used[a] = true
				out = append(out, m)
				continue
			}
		}
		used[a], used[base] = true, true
		out = append(out, &mnode{tex: a.accent + `{` + base.tex + `}`, x0: min(a.x0, base.x0), x1: max(a.x1, base.x1),
			top: min(a.top, base.top), bot: base.bot, base: base.base, size: base.size, built: base.built, big: base.big})
	}
	for _, n := range ns {
		if !used[n] {
			out = append(out, n)
		}
	}
	return out
}

// limitRow is the row of ns that a limit of o is: the parts end to end
// around the one under or over o's middle.
func limitRow(ns []*mnode, o *mnode, all []*mnode) []*mnode {
	sort.SliceStable(ns, func(i, j int) bool { return ns[i].x0 < ns[j].x0 })
	mid := (o.x0 + o.x1) / 2
	k := -1
	for i, n := range ns {
		if c := (n.x0 + n.x1) / 2; c > o.x0-0.1*o.size && c < o.x1+0.1*o.size && (k < 0 || math.Abs(c-mid) < math.Abs((ns[k].x0+ns[k].x1)/2-mid)) {
			k = i
		}
	}
	if k < 0 {
		return nil
	}
	// A part under another operator is that operator's limit.
	other := func(n *mnode) bool {
		c := (n.x0 + n.x1) / 2
		for _, p := range all {
			if p != o && p.limits && !stackable[p.tex] && p.size >= 0.9*o.size && c > p.x0 && c < p.x1 {
				return true
			}
		}
		return false
	}
	lo, hi := k, k
	for lo > 0 && ns[lo].x0-ns[lo-1].x1 < 0.4*o.size && !other(ns[lo-1]) {
		lo--
	}
	for hi+1 < len(ns) && ns[hi+1].x0-ns[hi].x1 < 0.4*o.size && !other(ns[hi+1]) {
		hi++
	}
	return ns[lo : hi+1]
}

// applyLimits puts the limits set under and over an operator on it: over
// big operators and function names first, then text over relations.
func applyLimits(ns []*mnode, relations bool) []*mnode {
	used := map[*mnode]bool{}
	var out []*mnode
	for _, o := range ns {
		if !o.limits || used[o] || stackable[o.tex] != relations {
			continue
		}
		var under, over []*mnode
		for _, n := range ns {
			if n == o || used[n] || n.big || n.size >= 0.95*o.size && !n.built ||
				n.x1 < o.x0-3*o.size || n.x0 > o.x1+3*o.size {
				continue
			}
			switch {
			case n.top >= o.bot-0.3*o.size && n.top < o.bot+o.size:
				under = append(under, n)
			case n.bot <= o.top+0.3*o.size && n.bot > o.top-o.size:
				over = append(over, n)
			}
		}
		under, over = limitRow(under, o, ns), limitRow(over, o, ns)
		if len(under) == 0 && len(over) == 0 {
			continue
		}
		m := &mnode{tex: o.tex, x0: o.x0, x1: o.x1, top: o.top, bot: o.bot, base: o.base, size: o.size, big: o.big}
		if strings.ContainsAny(m.tex, "_^") {
			m.tex = "{" + m.tex + "}"
		}
		if len(under) > 0 {
			m.tex += `_{` + arrange(under, o.size) + `}`
		}
		if len(over) > 0 {
			m.tex += `^{` + arrange(over, o.size) + `}`
		}
		used[o] = true
		for _, n := range append(under, over...) {
			used[n] = true
			m.x0, m.x1 = min(m.x0, n.x0), max(m.x1, n.x1)
			m.top, m.bot = min(m.top, n.top), max(m.bot, n.bot)
		}
		out = append(out, m)
	}
	for _, n := range ns {
		if !used[n] {
			out = append(out, n)
		}
	}
	return out
}

func wordTeX(w string) string {
	if opWords[w] && len(w) > 1 {
		switch w {
		case "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth",
			"log", "ln", "lg", "exp", "max", "min", "sup", "inf", "lim", "liminf", "limsup", "det", "dim", "ker",
			"deg", "gcd", "arg", "Pr", "hom":
			return `\` + w
		}
		return `\operatorname{` + w + `}`
	}
	return `\mathrm{` + w + `}`
}

// symbolTeX writes a symbol of a formula in LaTeX.
func symbolTeX(t, font string) string {
	var parts []string
	for _, r := range t {
		if r >= 0xFE00 && r <= 0xFE0F {
			continue // a variation selector
		}
		plain, variant := mathAlnum(r)
		if variant == "" {
			variant = fontVariant(font, plain)
		}
		var s string
		switch {
		case plain == 'ǫ' && mathFont(font):
			plain = 'ϵ' // TeX's \epsilon, mapped to Latin o with ogonek by glyphtounicode
		case plain == '\u033A' && mathFont(font):
			plain = 'ϱ' // and \varrho to a combining mark
		case unicode.Is(unicode.Mn, plain) && plain != '\u0338':
			continue // a lone combining mark is no LaTeX
		}
		if c, ok := mathCommands[plain]; ok {
			s = c
		} else if plain < 128 && strings.ContainsRune(`#$%&_{}`, plain) {
			s = `\` + string(plain)
		} else if plain == '\\' {
			s = `\backslash`
		} else {
			s = string(plain)
		}
		if variant != "" && plain < 128 && (unicode.IsLetter(plain) || unicode.IsDigit(plain)) {
			s = variant + `{` + s + `}`
		}
		parts = append(parts, s)
	}
	return joinTeX(parts)
}

// joinTeX writes LaTeX tokens one after the other, with a space only
// where a command's name would run on into a letter: dense, a formula with
// no markup reads as its text ("$=2,494$", not "$= 2 , 4 9 4$").
func joinTeX(ts []string) string {
	var sb strings.Builder
	for _, t := range ts {
		if t == "" {
			continue
		}
		if sb.Len() > 0 && endsWithCommand(sb.String()) && unicode.IsLetter([]rune(t)[0]) {
			sb.WriteByte(' ')
		}
		sb.WriteString(t)
	}
	return sb.String()
}

func endsWithCommand(s string) bool {
	i := len(s)
	for i > 0 && (s[i-1] >= 'a' && s[i-1] <= 'z' || s[i-1] >= 'A' && s[i-1] <= 'Z') {
		i--
	}
	return i < len(s) && i > 0 && s[i-1] == '\\'
}

// mathAlnum is the letter or digit that a character of Unicode's
// mathematical alphanumeric symbols stands for, and its alphabet.
func mathAlnum(r rune) (rune, string) {
	switch r {
	case 'ℝ', 'ℕ', 'ℤ', 'ℚ', 'ℂ', 'ℙ', 'ℍ':
		return map[rune]rune{'ℝ': 'R', 'ℕ': 'N', 'ℤ': 'Z', 'ℚ': 'Q', 'ℂ': 'C', 'ℙ': 'P', 'ℍ': 'H'}[r], `\mathbb`
	case 'ℒ', 'ℱ', 'ℋ', 'ℐ', 'ℛ', 'ℬ', 'ℰ', 'ℳ':
		return map[rune]rune{'ℒ': 'L', 'ℱ': 'F', 'ℋ': 'H', 'ℐ': 'I', 'ℛ': 'R', 'ℬ': 'B', 'ℰ': 'E', 'ℳ': 'M'}[r], `\mathcal`
	case 'ℭ', 'ℌ', 'ℨ':
		return map[rune]rune{'ℭ': 'C', 'ℌ': 'H', 'ℨ': 'Z'}[r], `\mathfrak`
	case 'ℎ':
		return 'h', ""
	}
	switch {
	case r >= 0x1D400 && r <= 0x1D6A3:
		i := int(r - 0x1D400)
		block, k := i/52, i%52
		letter := rune('A' + k)
		if k >= 26 {
			letter = rune('a' + k - 26)
		}
		variant := map[int]string{3: `\mathcal`, 4: `\mathcal`, 5: `\mathfrak`, 6: `\mathbb`, 7: `\mathfrak`}[block]
		return letter, variant
	case r >= 0x1D6A8 && r <= 0x1D7C9:
		k := int(r-0x1D6A8) % 58
		greek := []rune("ΑΒΓΔΕΖΗΘϴΙΚΛΜΝΞΟΠΡϴΣΤΥΦΧΨΩ∇αβγδεζηθικλμνξοπρςστυφχψω∂ϵϑϰϕϱϖ")
		if k < len(greek) {
			return greek[k], ""
		}
	case r >= 0x1D7CE && r <= 0x1D7FF:
		v := ""
		if r >= 0x1D7D8 && r <= 0x1D7E1 {
			v = `\mathbb`
		}
		return rune('0' + (r-0x1D7CE)%10), v
	}
	return r, ""
}

var mathCommands = map[rune]string{
	'α': `\alpha`, 'β': `\beta`, 'γ': `\gamma`, 'δ': `\delta`, 'ε': `\varepsilon`, 'ϵ': `\epsilon`, 'ζ': `\zeta`,
	'η': `\eta`, 'θ': `\theta`, 'ϑ': `\vartheta`, 'ι': `\iota`, 'κ': `\kappa`, 'ϰ': `\varkappa`, 'λ': `\lambda`,
	'μ': `\mu`, 'ν': `\nu`, 'ξ': `\xi`, 'π': `\pi`, 'ϖ': `\varpi`, 'ρ': `\rho`, 'ϱ': `\varrho`, 'σ': `\sigma`,
	'ς': `\varsigma`, 'τ': `\tau`, 'υ': `\upsilon`, 'φ': `\varphi`, 'ϕ': `\phi`, 'χ': `\chi`, 'ψ': `\psi`,
	'ω': `\omega`, 'Γ': `\Gamma`, 'Δ': `\Delta`, 'Θ': `\Theta`, 'Λ': `\Lambda`, 'Ξ': `\Xi`, 'Π': `\Pi`,
	'Σ': `\Sigma`, 'Υ': `\Upsilon`, 'Φ': `\Phi`, 'Ψ': `\Psi`, 'Ω': `\Omega`, 'ο': `o`,
	'≤': `\leq`, '≥': `\geq`, '⩽': `\leqslant`, '⩾': `\geqslant`, '≠': `\neq`, '≡': `\equiv`, '≈': `\approx`,
	'≃': `\simeq`, '≅': `\cong`, '∼': `\sim`, '∝': `\propto`, '≪': `\ll`, '≫': `\gg`, '≺': `\prec`, '≻': `\succ`,
	'⪯': `\preceq`, '⪰': `\succeq`, '≼': `\preccurlyeq`, '≽': `\succcurlyeq`, '∈': `\in`, '∉': `\notin`, '∋': `\ni`,
	'⊂': `\subset`, '⊃': `\supset`, '⊆': `\subseteq`, '⊇': `\supseteq`, '⊊': `\subsetneq`, '⊋': `\supsetneq`,
	'∪': `\cup`, '∩': `\cap`, '⊔': `\sqcup`, '⊓': `\sqcap`, '⊎': `\uplus`, '∖': `\setminus`, '∅': `\emptyset`,
	'∀': `\forall`, '∃': `\exists`, '∄': `\nexists`, '¬': `\neg`, '∧': `\wedge`, '∨': `\vee`, '⊕': `\oplus`,
	'⊗': `\otimes`, '⊙': `\odot`, '⊖': `\ominus`, '±': `\pm`, '∓': `\mp`, '×': `\times`, '÷': `\div`, '·': `\cdot`,
	'⋅': `\cdot`, '∘': `\circ`, '∗': `\ast`, '⋆': `\star`, '•': `\bullet`, '†': `\dagger`, '‡': `\ddagger`,
	'→': `\to`, '←': `\leftarrow`, '↔': `\leftrightarrow`, '⇒': `\Rightarrow`, '⇐': `\Leftarrow`,
	'⇔': `\Leftrightarrow`, '↦': `\mapsto`, '↪': `\hookrightarrow`, '↑': `\uparrow`, '↓': `\downarrow`,
	'⟶': `\longrightarrow`, '⟹': `\Longrightarrow`, '⇀': `\rightharpoonup`, '⇝': `\rightsquigarrow`,
	'∞': `\infty`, '∂': `\partial`, '∇': `\nabla`, 'ℓ': `\ell`, '℘': `\wp`, 'ı': `\imath`, 'ȷ': `\jmath`,
	'ℏ': `\hbar`, '′': `\prime`, '″': `\prime\prime`, '√': `\surd`, '∑': `\sum`, '∏': `\prod`, '∐': `\coprod`,
	'∫': `\int`, '∬': `\iint`, '∭': `\iiint`, '∮': `\oint`, '⋃': `\bigcup`, '⋂': `\bigcap`, '⨁': `\bigoplus`,
	'⨂': `\bigotimes`, '⨀': `\bigodot`, '⋁': `\bigvee`, '⋀': `\bigwedge`, '⨆': `\bigsqcup`, '⨄': `\biguplus`,
	'⟨': `\langle`, '⟩': `\rangle`, '⌊': `\lfloor`, '⌋': `\rfloor`, '⌈': `\lceil`, '⌉': `\rceil`, '‖': `\|`,
	'∥': `\parallel`, '⊥': `\perp`, '⊤': `\top`, '⊢': `\vdash`, '⊣': `\dashv`, '⊨': `\models`, '∣': `\mid`,
	'…': `\ldots`, '⋯': `\cdots`, '⋮': `\vdots`, '⋱': `\ddots`, '−': `-`, '∠': `\angle`, '△': `\triangle`,
	'□': `\square`, '◁': `\triangleleft`, '▷': `\triangleright`, '⋉': `\ltimes`, '⋊': `\rtimes`, '≍': `\asymp`,
	'≜': `\triangleq`, '≔': `\coloneqq`, '∤': `\nmid`, '≲': `\lesssim`, '≳': `\gtrsim`, '≐': `\doteq`,
	'⊸': `\multimap`, '♯': `\sharp`, '♭': `\flat`, '♮': `\natural`, '∡': `\measuredangle`,
	'◦': `\circ`, '\u0338': `\not`, 'ℜ': `\Re`, 'ℑ': `\Im`, '∆': `\Delta`, '\u2126': `\Omega`, 'µ': `\mu`, 'ℵ': `\aleph`, '〈': `\langle`, '〉': `\rangle`, '|': `|`, '<': `<`, '>': `>`, '~': `\sim`, '^': `\wedge`,
}

var (
	formulaSpan = regexp.MustCompile(`\$[^$]+\$`)
	// What a displayed formula's line may hold between its formulas: a
	// comma, a few words (\text{for all}).
	between = regexp.MustCompile(`^\s*[,;]?\s*((?:\pL+\s+){0,3}\pL+)?\s*[,;]?\s*$`)
	// And after its last: punctuation, an equation number.
	formulaEnd = regexp.MustCompile(`^\s*[.,;]?\s*(\(\d+(?:\.\d+)*[a-z]?\))?\s*$`)
)

// joinFormulaWords makes one formula of a line that is formulas with a few
// words between them ("$x > 0$ for all $x$"): a displayed formula with
// text in it.
func joinFormulaWords(line string) string {
	locs := formulaSpan.FindAllStringIndex(line, -1)
	if len(locs) < 2 || strings.TrimSpace(line[:locs[0][0]]) != "" || !formulaEnd.MatchString(line[locs[len(locs)-1][1]:]) {
		return line
	}
	var sb strings.Builder
	sb.WriteString("$")
	for i, l := range locs {
		if i > 0 {
			gap := line[locs[i-1][1]:l[0]]
			m := between.FindStringSubmatch(gap)
			if m == nil {
				return line
			}
			if strings.Contains(gap, ",") {
				sb.WriteString(" ,")
			}
			if m[1] != "" {
				sb.WriteString(` \text{ ` + m[1] + ` }`)
			}
			sb.WriteString(" ")
		}
		sb.WriteString(line[l[0]+1 : l[1]-1])
	}
	sb.WriteString("$")
	sb.WriteString(line[locs[len(locs)-1][1]:])
	return sb.String()
}

// pureFormula is the formula a line holds and nothing else, but for
// punctuation and an equation number after it.
func pureFormula(line string) (tex, tail string, ok bool) {
	l := formulaSpan.FindAllStringIndex(line, -1)
	if len(l) != 1 || strings.TrimSpace(line[:l[0][0]]) != "" || !formulaEnd.MatchString(line[l[0][1]:]) {
		return "", "", false
	}
	return line[l[0][0]+1 : l[0][1]-1], line[l[0][1]:], true
}

// joinFormulaRows makes one formula of lines that are each a formula, one
// under the other: the rows of a displayed formula (\begin{aligned}).
func joinFormulaRows(boxes []Box, lines []Line) ([]Box, []Line) {
	var outB []Box
	var outL []Line
	for i := 0; i < len(lines); i++ {
		tex, tail, ok := pureFormula(lines[i].Text)
		if !ok {
			outB, outL = append(outB, boxes[i]), append(outL, lines[i])
			continue
		}
		rows, tails, b := []string{tex}, []string{tail}, boxes[i]
		j := i + 1
		for ; j < len(lines); j++ {
			t, tl, ok := pureFormula(lines[j].Text)
			o := boxes[j]
			h := min(b.Y1-b.Y0, o.Y1-o.Y0)
			if !ok || o.Y0-b.Y1 > 0.8*h || o.Y0 < b.Y1-0.5*h || o.X0 > b.X1 || o.X1 < b.X0 {
				break
			}
			rows, tails = append(rows, t), append(tails, tl)
			b = Box{min(b.X0, o.X0), min(b.Y0, o.Y0), max(b.X1, o.X1), max(b.Y1, o.Y1)}
		}
		if len(rows) == 1 {
			outB, outL = append(outB, boxes[i]), append(outL, lines[i])
			continue
		}
		text := `$\begin{aligned} ` + strings.Join(rows, ` \\ `) + ` \end{aligned}$`
		for _, tl := range tails {
			if t := strings.TrimSpace(tl); t != "" {
				text += " " + t
			}
		}
		outB, outL = append(outB, b), append(outL, Line{Text: text, Confidence: 1})
		i = j - 1
	}
	return outB, outL
}
