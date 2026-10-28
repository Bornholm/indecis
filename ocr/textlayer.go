package ocr

import (
	"fmt"
	"image"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/bornholm/indecis/internal/pdf"
)

// A PDF is an open PDF file, whose pages' text layers can be read instead
// of recognized: a born-digital page holds its text, with no reading
// errors. The pages themselves are not rendered: reading a page without a
// sound text layer (a scan) takes an image of it.
type PDF struct {
	pages []*pdf.Page
	// Math writes the formulas of the text layers as LaTeX between
	// dollars ($\frac{1}{2}$), where they are set with math fonts.
	Math bool
}

// OpenPDF reads the structure of a PDF file.
func OpenPDF(data []byte) (*PDF, error) {
	f, err := pdf.Open(data)
	if err != nil {
		return nil, err
	}
	return &PDF{pages: f.Pages()}, nil
}

// NumPages is the number of pages of the file.
func (p *PDF) NumPages() int { return len(p.pages) }

// The checks that decide whether a text layer is the page's text, as Indra
// OCR's: an OCR layer over a scan is drawn invisibly, a broken font map
// gives characters with no text, text with no ink under it is hidden or
// misplaced.
const (
	maxInvisible = 0.05  // share of invisible characters
	maxBadChars  = 0.005 // share of characters with no text
	maxUnmapped  = 0.05  // share of named glyphs that are no known character
	maxNoInk     = 0.03  // share of lines with no ink under them
	inkLevel     = 160   // grey level below which a pixel is ink
)

// A TextLayer is the text of a page as its PDF draws it, in lines.
type TextLayer struct {
	Width, Height float64 // the page as shown, in points
	Turns         int     // quarter turns (clockwise) of the page that set most of its text upright
	Boxes         []Box   // the lines, in points on the page turned upright
	Lines         []Line
	Chars         int    // characters drawn, spaces excluded
	Trusted       bool   // the layer passed the checks: it is the page's text
	Reason        string // why it did not
	// OCRLayer: the text is drawn invisibly over a scan, by the OCR that
	// made the file. Its lines are there, not trusted: TrustOCR uses them.
	OCRLayer bool
	bad      string // the layer's font map is broken
}

// TrustOCR uses an OCR layer as the page's text when it passes the other
// checks: an OCR engine's reading of the scan, which may be better or
// worse than indecis's own.
func (t *TextLayer) TrustOCR() {
	if t.OCRLayer && t.bad == "" && len(t.Lines) > 0 {
		t.Trusted, t.Reason = true, ""
	}
}

// TextLayer reads the text layer of page i (from 0) and checks it.
func (p *PDF) TextLayer(i int) (*TextLayer, error) {
	if i < 0 || i >= len(p.pages) {
		return nil, fmt.Errorf("pdf: no page %d", i+1)
	}
	pg := p.pages[i]
	w, h := pg.Size()
	glyphs, rules := pg.Text()
	if !p.Math {
		rules = nil
	}
	return layerLines(glyphs, rules, w, h, p.Math), nil
}

// layerGlyph is a glyph in the frame of its line: u along the baseline, v
// across it, downwards.
type layerGlyph struct {
	text           string
	u0, v0, u1, v1 float64
	size           float64
	space          bool
}

type layerLine struct {
	angle  int // the baseline's direction on the page, in degrees
	glyphs []layerGlyph
	u0, v0 float64
	u1, v1 float64
	size   float64
}

func layerLines(glyphs []pdf.Glyph, rules []pdf.Rule, w, h float64, formulas bool) *TextLayer {
	t := &TextLayer{Width: w, Height: h}
	invisible, bad, unmapped := 0, 0, 0
	quarters := [4]int{}
	for _, g := range glyphs {
		if g.Space {
			continue
		}
		t.Chars++
		if g.Invisible {
			invisible++
		}
		switch {
		case !g.Known || g.Text != "" && badText(g.Text):
			bad++
		case g.Text == "" || privateUse(g.Text):
			unmapped++
		}
		quarters[quarter(g.Dx, g.Dy)]++
	}
	switch {
	case t.Chars == 0:
		t.Reason = "no text layer"
		return t
	case float64(bad) > maxBadChars*float64(t.Chars) || float64(unmapped) > maxUnmapped*float64(t.Chars):
		t.Reason = fmt.Sprintf("broken font map (%d characters with no text, %d unknown glyphs)", bad, unmapped)
		t.bad = t.Reason
	case float64(invisible) > maxInvisible*float64(t.Chars):
		t.Reason = fmt.Sprintf("invisible text %.0f%% (an OCR layer)", 100*float64(invisible)/float64(t.Chars))
		t.OCRLayer = true
	default:
		t.Trusted = true
	}
	// A page set sideways: most of its text runs up or down.
	for q := 1; q < 4; q++ {
		if 2*quarters[q] > t.Chars {
			t.Turns = (4 - q) % 4
		}
	}
	if t.Turns%2 == 1 {
		t.Width, t.Height = h, w
	}
	if formulas && t.Turns == 0 {
		glyphs = mathGlyphs(glyphs, rules)
	}
	var lines []*layerLine
	var cur *layerLine
	for _, g := range glyphs {
		if g.Invisible && !t.OCRLayer || !g.Space && (!g.Known || badText(g.Text) || privateUse(g.Text)) {
			// Glyphs with no text are dropped: a math delimiter's pieces.
			continue
		}
		x0, y0, x1, y1 := turnBox(g.X0, g.Y0, g.X1, g.Y1, w, h, t.Turns)
		dx, dy := turnDir(g.Dx, g.Dy, t.Turns)
		if x1 < 0 || y1 < 0 || x0 > t.Width || y0 > t.Height {
			continue // off the page
		}
		angle := int(math.Round(math.Atan2(dy, dx) * 180 / math.Pi))
		// The glyph box in the frame of its baseline.
		u0, v0, u1, v1 := frameBox(x0, y0, x1, y1, dx, dy)
		lg := layerGlyph{text: g.Text, u0: u0, v0: v0, u1: u1, v1: v1, size: max(g.Size, 0.1), space: g.Space}
		if cur != nil && cur.accepts(lg, angle) {
			cur.add(lg)
			continue
		}
		if g.Space {
			continue // a space starts no line
		}
		cur = &layerLine{angle: angle, u0: u0, v0: v0, u1: u1, v1: v1, size: lg.size}
		cur.glyphs = []layerGlyph{lg}
		lines = append(lines, cur)
	}
	lines = splitGutters(mergeLines(lines))
	for _, l := range lines {
		text := l.text()
		if strings.TrimSpace(text) == "" {
			continue
		}
		if formulas {
			text = joinFormulaWords(text)
		}
		t.Boxes = append(t.Boxes, l.box())
		t.Lines = append(t.Lines, Line{Text: text, Confidence: 1})
	}
	if formulas {
		t.Boxes, t.Lines = joinFormulaRows(t.Boxes, t.Lines)
	}
	return t
}

// badText reports text that is no text: empty, a replacement character or
// a control character.
func badText(s string) bool {
	if s == "" {
		return true
	}
	for _, r := range s {
		if r == unicode.ReplacementChar || unicode.IsControl(r) && r != '\t' {
			return true
		}
	}
	return false
}

// privateUse reports text in Unicode's private use area: a symbol font's
// codes (Wingdings, pieces of math delimiters), no characters.
func privateUse(s string) bool {
	for _, r := range s {
		if unicode.Is(unicode.Co, r) {
			return true
		}
	}
	return false
}

// quarter is the quarter turn of a baseline direction: 0 left to right,
// 1 top to bottom, 2 right to left, 3 bottom to top.
func quarter(dx, dy float64) int {
	a := math.Atan2(dy, dx) / (math.Pi / 2)
	return ((int(math.Round(a)) % 4) + 4) % 4
}

// turnBox maps a box of a w×h page to the page turned clockwise by turns
// quarter turns.
func turnBox(x0, y0, x1, y1, w, h float64, turns int) (float64, float64, float64, float64) {
	switch turns {
	case 1:
		return h - y1, x0, h - y0, x1
	case 2:
		return w - x1, h - y1, w - x0, h - y0
	case 3:
		return y0, w - x1, y1, w - x0
	}
	return x0, y0, x1, y1
}

func turnDir(dx, dy float64, turns int) (float64, float64) {
	for range turns {
		dx, dy = -dy, dx
	}
	return dx, dy
}

// frameBox is the box of the corners of (x0, y0, x1, y1) in the frame of
// the baseline direction (dx, dy).
func frameBox(x0, y0, x1, y1, dx, dy float64) (u0, v0, u1, v1 float64) {
	u0, v0, u1, v1 = math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
	for _, c := range [4][2]float64{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}} {
		u := c[0]*dx + c[1]*dy
		v := -c[0]*dy + c[1]*dx
		u0, u1 = min(u0, u), max(u1, u)
		v0, v1 = min(v0, v), max(v1, v)
	}
	return
}

// gapSplit is the widest gap within a line, in ems: wider gaps separate
// columns or table cells, as the detector cuts lines at wide gaps.
const gapSplit = 1.5

// accepts reports whether g goes on the line: same direction, about the
// same baseline, and next along it (a little back for kerning and
// overprinted accents).
func (l *layerLine) accepts(g layerGlyph, angle int) bool {
	if angle != l.angle {
		return false
	}
	h := min(l.v1-l.v0, g.v1-g.v0)
	if g.space {
		return g.u0 >= l.u1-0.5*l.size && g.u0-l.u1 < gapSplit*l.size
	}
	overlap := min(l.v1, g.v1) - max(l.v0, g.v0)
	if overlap < sameLine(l.size, g.size)*h {
		return false
	}
	last := l.glyphs[len(l.glyphs)-1]
	return g.u0 >= last.u0-0.3*l.size && g.u0-l.u1 < gapSplit*max(l.size, g.size)
}

// sameLine is the vertical overlap, × the smaller height, from which
// glyphs of sizes a and b are on one line: less for a sub- or superscript.
func sameLine(a, b float64) float64 {
	if min(a, b) < 0.85*max(a, b) {
		return 0.25
	}
	return 0.5
}

func (l *layerLine) add(g layerGlyph) {
	l.glyphs = append(l.glyphs, g)
	if g.space {
		return
	}
	l.u0, l.u1 = min(l.u0, g.u0), max(l.u1, g.u1)
	// The line keeps the height of its main text: a superscript does not
	// stretch it.
	if g.size >= 0.8*l.size {
		l.v0, l.v1 = min(l.v0, g.v0), max(l.v1, g.v1)
	}
	l.size = max(l.size, g.size)
}

// mergeLines joins the pieces of a line drawn apart (a page that draws
// word by word, or a line's end after a footnote mark) along each baseline.
func mergeLines(lines []*layerLine) []*layerLine {
	sort.SliceStable(lines, func(i, j int) bool {
		if lines[i].angle != lines[j].angle {
			return lines[i].angle < lines[j].angle
		}
		return lines[i].u0 < lines[j].u0
	})
	var out []*layerLine
	for _, l := range lines {
		merged := false
		for k := len(out) - 1; k >= 0 && k >= len(out)-200; k-- {
			o := out[k]
			if o.angle != l.angle {
				break
			}
			h := min(o.v1-o.v0, l.v1-l.v0)
			overlap := min(o.v1, l.v1) - max(o.v0, l.v0)
			gap := l.u0 - o.u1
			if overlap >= sameLine(o.size, l.size)*h && gap >= -0.3*h && gap < gapSplit*max(o.size, l.size) {
				for _, g := range l.glyphs {
					o.add(g)
				}
				o.u0 = min(o.u0, l.u0)
				merged = true
				break
			}
		}
		if !merged {
			out = append(out, l)
		}
	}
	return out
}

// splitGutters cuts lines across a narrow column gutter: at a gap of at
// least gutterEm and twice the line's gap between words, when at least
// gutterLines lines near it are blank there too.
func splitGutters(lines []*layerLine) []*layerLine {
	const (
		gutterEm    = 0.8
		gutterLines = 3
	)
	var out []*layerLine
	for i, l := range lines {
		ink := l.ink()
		word := medianWordGap(ink, l.size)
		start := 0
		var pieces []*layerLine
		for k := 1; k < len(ink); k++ {
			a, b := ink[k-1], ink[k]
			gap := b.u0 - a.u1
			if gap < gutterEm*l.size || gap < 2*word {
				continue
			}
			c, w := (a.u1+b.u0)/2, gap/4
			n := 0
			for j, o := range lines {
				if j == i || o.angle != l.angle || abs(o.v0-l.v0) > 12*l.size || o.u1 < c-2*l.size-gap || o.u0 > c+2*l.size+gap {
					continue
				}
				if !o.inkWithin(c-w, c+w) {
					n++
				}
			}
			if n >= gutterLines {
				pieces = append(pieces, l.piece(ink[start:k]))
				start = k
			}
		}
		if start == 0 {
			out = append(out, l)
			continue
		}
		out = append(out, append(pieces, l.piece(ink[start:]))...)
	}
	return out
}

// ink is the line's glyphs other than spaces, with the spaces that follow
// each kept in its piece by piece.
func (l *layerLine) ink() []layerGlyph {
	var gs []layerGlyph
	for _, g := range l.glyphs {
		if !g.space {
			gs = append(gs, g)
		}
	}
	return gs
}

func medianWordGap(ink []layerGlyph, size float64) float64 {
	var gaps []float64
	for k := 1; k < len(ink); k++ {
		if gap := ink[k].u0 - ink[k-1].u1; gap > spaceGap*size {
			gaps = append(gaps, gap)
		}
	}
	if len(gaps) == 0 {
		return 0
	}
	sort.Float64s(gaps)
	return gaps[len(gaps)/2]
}

func (l *layerLine) inkWithin(a, b float64) bool {
	for _, g := range l.glyphs {
		if !g.space && g.u1 > a && g.u0 < b {
			return true
		}
	}
	return false
}

func (l *layerLine) piece(gs []layerGlyph) *layerLine {
	p := &layerLine{angle: l.angle, u0: gs[0].u0, v0: l.v0, u1: gs[0].u1, v1: l.v1, size: l.size}
	for _, g := range gs {
		p.glyphs = append(p.glyphs, g)
		p.u0, p.u1 = min(p.u0, g.u0), max(p.u1, g.u1)
	}
	return p
}

func (l *layerLine) box() Box {
	// Back from the line's frame to the page.
	a := float64(l.angle) * math.Pi / 180
	dx, dy := math.Cos(a), math.Sin(a)
	x0, y0, x1, y1 := math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
	for _, c := range [4][2]float64{{l.u0, l.v0}, {l.u1, l.v0}, {l.u0, l.v1}, {l.u1, l.v1}} {
		x := c[0]*dx - c[1]*dy
		y := c[0]*dy + c[1]*dx
		x0, x1 = min(x0, x), max(x1, x)
		y0, y1 = min(y0, y), max(y1, y)
	}
	return Box{x0, y0, x1, y1}
}

// spaceGap is the gap between two glyphs, in ems, from which they are
// separate words: the text layers of TeX and others draw no spaces.
const spaceGap = 0.15

func (l *layerLine) text() string {
	type token struct {
		text   string
		u1     float64
		size   float64
		g      *layerGlyph
		spaced bool // a space comes before it
	}
	var toks []token
	space := false
	for i := range l.glyphs {
		g := &l.glyphs[i]
		if g.space {
			space = true
			continue
		}
		if n := len(toks); n > 0 && !space {
			if c, ok := accent(toks[n-1].g, g); ok {
				// An accent drawn as a glyph of its own, over its letter.
				p := &toks[n-1]
				p.text, p.u1, p.g = c, max(p.u1, g.u1), g
				continue
			}
		}
		toks = append(toks, token{text: g.text, u1: g.u1, size: g.size, g: g, spaced: space})
		space = false
	}
	var sb strings.Builder
	for i, t := range toks {
		if i > 0 {
			p := toks[i-1]
			if t.spaced || t.g.u0-p.u1 > spaceGap*min(p.size, t.size) || strings.HasSuffix(p.text, "$") && strings.HasPrefix(t.text, "$") {
				sb.WriteByte(' ')
			}
		}
		sb.WriteString(t.text)
	}
	return NormalizeText(Compose(expandLigatures(sb.String())))
}

var combining = map[string]rune{
	"´": '\u0301', "`": '\u0300', "^": '\u0302', "ˆ": '\u0302', "¨": '\u0308', "˜": '\u0303', "~": '\u0303',
	"¸": '\u0327', "ˇ": '\u030C', "˘": '\u0306', "˙": '\u0307', "˚": '\u030A', "˝": '\u030B', "˛": '\u0328', "¯": '\u0304',
	"\u0301": '\u0301', "\u0300": '\u0300', "\u0302": '\u0302', "\u0308": '\u0308', "\u0303": '\u0303', "\u0327": '\u0327',
	"\u030C": '\u030C', "\u0306": '\u0306', "\u0307": '\u0307', "\u030A": '\u030A', "\u030B": '\u030B', "\u0328": '\u0328', "\u0304": '\u0304',
}

// accent combines a spacing accent with the letter it is drawn over (TeX
// draws "é" as "´" and "e"), in either order.
func accent(a, b *layerGlyph) (string, bool) {
	m, ok := combining[a.text]
	letter := b
	if !ok {
		if m, ok = combining[b.text]; !ok {
			return "", false
		}
		a, letter = b, a
	}
	rs := []rune(letter.text)
	if len(rs) != 1 || !unicode.IsLetter(rs[0]) {
		return "", false
	}
	w := a.u1 - a.u0
	overlap := min(a.u1, letter.u1) - max(a.u0, letter.u0)
	if w <= 0 || overlap < 0.5*w {
		return "", false
	}
	return Compose(letter.text + string(m)), true
}

var ligatures = strings.NewReplacer("ﬀ", "ff", "ﬁ", "fi", "ﬂ", "fl", "ﬃ", "ffi", "ﬄ", "ffl", "ﬅ", "st", "ﬆ", "st")

func expandLigatures(s string) string { return ligatures.Replace(s) }

// Scaled returns the layer's lines with their boxes in the pixels of an
// image of the page turned upright, w×h pixels.
func (t *TextLayer) Scaled(w, h int) ([]Box, []Line) {
	sx, sy := float64(w)/t.Width, float64(h)/t.Height
	boxes := make([]Box, len(t.Boxes))
	for i, b := range t.Boxes {
		boxes[i] = Box{b.X0 * sx, b.Y0 * sy, b.X1 * sx, b.Y1 * sy}
	}
	lines := append([]Line(nil), t.Lines...)
	return boxes, lines
}

// CheckInk distrusts the layer when too many of its lines have no ink
// under them in page, an image of the page turned upright.
func (t *TextLayer) CheckInk(page *image.Gray) {
	if !t.Trusted || len(t.Boxes) == 0 {
		return
	}
	boxes, lines := t.Scaled(page.Rect.Dx(), page.Rect.Dy())
	empty := 0
	for i, b := range boxes {
		// A line of accents or pieces of symbols (a wide hat over a
		// formula) is where the font's metrics say, not where its ink is.
		if alnum(lines[i].Text) && !hasInk(page, b) {
			empty++
		}
	}
	if float64(empty) > maxNoInk*float64(len(boxes)) {
		t.Trusted = false
		t.Reason = fmt.Sprintf("%d of %d lines have no ink", empty, len(boxes))
	}
}

func alnum(s string) bool {
	for _, r := range s {
		if unicode.In(r, unicode.Lu, unicode.Ll, unicode.Lt, unicode.Lo, unicode.Nd) {
			return true
		}
	}
	return false
}

func hasInk(page *image.Gray, b Box) bool {
	r := image.Rect(int(b.X0), int(b.Y0), int(math.Ceil(b.X1))+1, int(math.Ceil(b.Y1))+1).Intersect(page.Rect)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		row := page.Pix[(y-page.Rect.Min.Y)*page.Stride:]
		for x := r.Min.X; x < r.Max.X; x++ {
			if row[x-page.Rect.Min.X] < inkLevel {
				return true
			}
		}
	}
	return false
}

// ReadLayerPage reads a page from its text layer and page, an image of it
// turned upright (t.Turns). With a detector and a recognizer, the lines
// the detector finds where the layer has no text (text in a picture) are
// read too. The lines come in reading order.
func ReadLayerPage(t *TextLayer, page *image.Gray, d *Detector, r *Recognizer) ([]Box, []Line) {
	boxes, lines := t.Scaled(page.Rect.Dx(), page.Rect.Dy())
	if d != nil && r != nil {
		var extra []Box
		for _, b := range d.Detect(page) {
			if covered(b, boxes) < 0.2 {
				extra = append(extra, b)
			}
		}
		for i, l := range r.ReadLines(page, extra) {
			boxes = append(boxes, extra[i])
			lines = append(lines, l)
		}
	}
	return Clean(boxes, lines)
}

// covered is the share of b's area that boxes cover (overlaps counted
// once per box).
func covered(b Box, boxes []Box) float64 {
	area := (b.X1 - b.X0) * (b.Y1 - b.Y0)
	if area <= 0 {
		return 1
	}
	s := 0.0
	for _, o := range boxes {
		w := min(b.X1, o.X1) - max(b.X0, o.X0)
		h := min(b.Y1, o.Y1) - max(b.Y0, o.Y0)
		if w > 0 && h > 0 {
			s += w * h
		}
	}
	return min(1, s/area)
}
