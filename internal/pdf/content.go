package pdf

import (
	"bytes"
	"math"
	"strings"
)

// A Glyph is one character drawn on a page.
type Glyph struct {
	Text           string  // what the glyph stands for; "" when unknown
	X0, Y0, X1, Y1 float64 // its box on the page as shown: points, origin top left
	Dx, Dy         float64 // the direction of its baseline on the page (unit vector)
	Size           float64 // the font size on the page, in points
	Known          bool    // the font maps the glyph to text
	Invisible      bool    // drawn in render mode 3 or 7 (an OCR layer over a scan)
	Space          bool    // a space: a word break, drawn or not
	Font           string  // the font's name (CMMI10, Times-Roman), subset tag left out
	Name           string  // the glyph's name, in a simple font (parenleftbig)
	OX, OY         float64 // the glyph's origin on its baseline, on the page as shown
	Advance        float64 // how far along the baseline the glyph moves, in points
}

// A Rule is a filled rectangle or a stroked straight line, as the page
// draws a fraction bar, the bar of a root or a table's border: its box on
// the page as shown.
type Rule struct {
	X0, Y0, X1, Y1 float64
}

type matrix [6]float64

var identity = matrix{1, 0, 0, 1, 0, 0}

// mul is m then n.
func (m matrix) mul(n matrix) matrix {
	return matrix{
		m[0]*n[0] + m[1]*n[2], m[0]*n[1] + m[1]*n[3],
		m[2]*n[0] + m[3]*n[2], m[2]*n[1] + m[3]*n[3],
		m[4]*n[0] + m[5]*n[2] + n[4], m[4]*n[1] + m[5]*n[3] + n[5],
	}
}

func (m matrix) apply(x, y float64) (float64, float64) {
	return m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5]
}

type gstate struct {
	ctm                   matrix
	font                  *font
	size                  float64
	charSp, wordSp, scale float64
	leading, rise         float64
	render                int
	lineW                 float64
}

type interp struct {
	p       *Page
	fonts   map[any]*font
	glyphs  []Glyph
	gs      gstate
	stack   []gstate
	tm, tlm matrix
	marks   []mark
	depth   int
	ops     int
	rules   []Rule
	subs    [][][2]float64 // the current path's subpaths, points on the page as shown
	curved  bool           // the current path has curves
}

type mark struct {
	start  int
	actual string
	has    bool
}

const (
	maxGlyphs = 500000
	maxOps    = 5000000
)

// Glyphs returns the characters drawn on the page, in the order the page
// draws them. Characters from form XObjects are included; marked content
// with an /ActualText replaces the glyphs it covers.
func (p *Page) Glyphs() []Glyph {
	g, _ := p.Text()
	return g
}

// Text returns the page's glyphs, as Glyphs, and the rules it draws: thin
// filled rectangles and straight stroked lines, at most maxRuleWidth thick.
func (p *Page) Text() ([]Glyph, []Rule) {
	in := &interp{p: p, fonts: map[any]*font{}}
	in.gs = gstate{ctm: identity, scale: 1, lineW: 1}
	in.run(p.contents(), p.resources)
	return in.glyphs, in.rules
}

// maxRuleWidth is the thickest rule kept, in points.
const maxRuleWidth = 3

const maxRules = 20000

func (in *interp) run(data []byte, res Dict) {
	f := in.p.f
	l := &lexer{b: data}
	var args []any
	for {
		t := l.token()
		if t == nil {
			return
		}
		in.ops++
		if in.ops > maxOps || len(in.glyphs) > maxGlyphs {
			return
		}
		op, ok := t.(keyword)
		if !ok {
			args = append(args, t)
			continue
		}
		if op == "[" || op == "<<" {
			args = append(args, l.build(t, 0))
			continue
		}
		switch op {
		case "q":
			in.stack = append(in.stack, in.gs)
		case "Q":
			if n := len(in.stack); n > 0 {
				in.gs = in.stack[n-1]
				in.stack = in.stack[:n-1]
			}
		case "cm":
			if m, ok := nums6(args); ok {
				in.gs.ctm = m.mul(in.gs.ctm)
			}
		case "w":
			in.gs.lineW = num(args, 0)
		case "m":
			in.subs = append(in.subs, [][2]float64{in.point(num(args, 0), num(args, 1))})
		case "l":
			if n := len(in.subs); n > 0 {
				in.subs[n-1] = append(in.subs[n-1], in.point(num(args, 0), num(args, 1)))
			}
		case "c", "v", "y":
			in.curved = true
		case "re":
			x, y, w, h := num(args, 0), num(args, 1), num(args, 2), num(args, 3)
			in.subs = append(in.subs, [][2]float64{in.point(x, y), in.point(x+w, y), in.point(x+w, y+h), in.point(x, y+h), in.point(x, y)})
		case "f", "F", "f*", "B", "B*", "b", "b*":
			in.fillRules()
			in.subs, in.curved = in.subs[:0], false
		case "S", "s":
			in.strokeRules()
			in.subs, in.curved = in.subs[:0], false
		case "n":
			in.subs, in.curved = in.subs[:0], false
		case "BT":
			in.tm, in.tlm = identity, identity
		case "ET":
		case "Tc":
			in.gs.charSp = num(args, 0)
		case "Tw":
			in.gs.wordSp = num(args, 0)
		case "Tz":
			in.gs.scale = num(args, 0) / 100
		case "TL":
			in.gs.leading = num(args, 0)
		case "Ts":
			in.gs.rise = num(args, 0)
		case "Tr":
			in.gs.render = int(num(args, 0))
		case "Tf":
			if len(args) >= 2 {
				name, _ := args[0].(Name)
				in.gs.font = in.font(f.dict(res["Font"])[name])
				in.gs.size = num(args, 1)
			}
		case "gs":
			if len(args) >= 1 {
				name, _ := args[0].(Name)
				ext := f.dict(f.dict(res["ExtGState"])[name])
				if fa := f.array(ext["Font"]); len(fa) == 2 {
					in.gs.font = in.font(fa[0])
					in.gs.size, _ = f.num(fa[1])
				}
			}
		case "Td":
			in.tlm = matrix{1, 0, 0, 1, num(args, 0), num(args, 1)}.mul(in.tlm)
			in.tm = in.tlm
		case "TD":
			in.gs.leading = -num(args, 1)
			in.tlm = matrix{1, 0, 0, 1, num(args, 0), num(args, 1)}.mul(in.tlm)
			in.tm = in.tlm
		case "Tm":
			if m, ok := nums6(args); ok {
				in.tm, in.tlm = m, m
			}
		case "T*":
			in.nextLine()
		case "Tj":
			if len(args) >= 1 {
				s, _ := args[len(args)-1].(String)
				in.show(s)
			}
		case "'":
			in.nextLine()
			if len(args) >= 1 {
				s, _ := args[len(args)-1].(String)
				in.show(s)
			}
		case "\"":
			if len(args) >= 3 {
				in.gs.wordSp = num(args, 0)
				in.gs.charSp = num(args, 1)
				in.nextLine()
				s, _ := args[2].(String)
				in.show(s)
			}
		case "TJ":
			if len(args) >= 1 {
				a, _ := args[len(args)-1].(Array)
				for _, v := range a {
					switch x := v.(type) {
					case String:
						in.show(x)
					case int64:
						in.adjust(float64(x))
					case float64:
						in.adjust(x)
					}
				}
			}
		case "Do":
			if len(args) >= 1 && in.depth < 12 {
				name, _ := args[0].(Name)
				if s, ok := f.Resolve(f.dict(res["XObject"])[name]).(*Stream); ok && f.name(s.Dict["Subtype"]) == "Form" {
					in.form(s, res)
				} else if ok && f.name(s.Dict["Subtype"]) == "Image" {
					in.imageRule()
				}
			}
		case "BMC":
			in.marks = append(in.marks, mark{start: len(in.glyphs)})
		case "BDC":
			m := mark{start: len(in.glyphs)}
			if len(args) >= 2 {
				var props Dict
				switch v := args[1].(type) {
				case Dict:
					props = v
				case Name:
					props = f.dict(f.dict(res["Properties"])[v])
				}
				if s, ok := f.Resolve(props["ActualText"]).(String); ok {
					m.actual, m.has = textString([]byte(s)), true
				}
			}
			in.marks = append(in.marks, m)
		case "EMC":
			if n := len(in.marks); n > 0 {
				m := in.marks[n-1]
				in.marks = in.marks[:n-1]
				if m.has {
					in.replace(m)
				}
			}
		case "BI":
			// An inline image: its data runs from ID to EI.
			in.imageRule()
			i := bytes.Index(data[l.pos:], []byte("ID"))
			if i < 0 {
				return
			}
			p := l.pos + i + 3
			for p+2 <= len(data) {
				j := bytes.Index(data[p:], []byte("EI"))
				if j < 0 {
					return
				}
				p += j
				before := p == 0 || isSpace(data[p-1])
				after := p+2 == len(data) || isSpace(data[p+2])
				p += 2
				if before && after {
					break
				}
			}
			l.pos = p
		}
		args = args[:0]
	}
}

// point maps a point of user space to the page as shown.
func (in *interp) point(x, y float64) [2]float64 {
	vx, vy := in.p.toView(in.gs.ctm.apply(x, y))
	return [2]float64{vx, vy}
}

// fillRules keeps the thin rectangles of a filled path: subpaths whose
// points are all corners of their box.
func (in *interp) fillRules() {
	if in.curved {
		return
	}
	for _, sp := range in.subs {
		if len(sp) < 4 || len(sp) > 5 {
			continue
		}
		x0, y0, x1, y1 := sp[0][0], sp[0][1], sp[0][0], sp[0][1]
		for _, q := range sp {
			x0, y0, x1, y1 = min(x0, q[0]), min(y0, q[1]), max(x1, q[0]), max(y1, q[1])
		}
		corner := true
		for _, q := range sp {
			if !(near(q[0], x0) || near(q[0], x1)) || !(near(q[1], y0) || near(q[1], y1)) {
				corner = false
			}
		}
		if corner {
			in.addRule(Rule{x0, y0, x1, y1})
		}
	}
}

// strokeRules keeps the straight level or upright segments of a stroked
// path, as wide as the line.
func (in *interp) strokeRules() {
	if in.curved {
		return
	}
	m := in.gs.ctm
	half := in.gs.lineW * math.Sqrt(math.Abs(m[0]*m[3]-m[1]*m[2])) / 2
	for _, sp := range in.subs {
		for k := 1; k < len(sp); k++ {
			a, b := sp[k-1], sp[k]
			switch {
			case near(a[1], b[1]):
				in.addRule(Rule{min(a[0], b[0]), a[1] - half, max(a[0], b[0]), a[1] + half})
			case near(a[0], b[0]):
				in.addRule(Rule{a[0] - half, min(a[1], b[1]), a[0] + half, max(a[1], b[1])})
			}
		}
	}
}

// imageRule keeps an image drawn as a thin bar: some files draw rules as
// an image of one pixel, stretched.
func (in *interp) imageRule() {
	var xs, ys [4]float64
	for i, c := range [4][2]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}} {
		p := in.point(c[0], c[1])
		xs[i], ys[i] = p[0], p[1]
	}
	in.addRule(Rule{min(xs[0], xs[1], xs[2], xs[3]), min(ys[0], ys[1], ys[2], ys[3]), max(xs[0], xs[1], xs[2], xs[3]), max(ys[0], ys[1], ys[2], ys[3])})
}

func (in *interp) addRule(r Rule) {
	w, h := r.X1-r.X0, r.Y1-r.Y0
	if min(w, h) <= maxRuleWidth && max(w, h) >= 2*min(w, h) && len(in.rules) < maxRules {
		in.rules = append(in.rules, r)
	}
}

func near(a, b float64) bool { return math.Abs(a-b) < 0.01 }

func (in *interp) form(s *Stream, res Dict) {
	f := in.p.f
	data, err := f.decode(s)
	if err != nil && len(data) == 0 {
		return
	}
	saved, stack := in.gs, len(in.stack)
	if m := f.array(s.Dict["Matrix"]); len(m) == 6 {
		var mm matrix
		ok := true
		for i := range 6 {
			mm[i], ok = f.num(m[i])
			if !ok {
				break
			}
		}
		if ok {
			in.gs.ctm = mm.mul(in.gs.ctm)
		}
	}
	if r := f.dict(s.Dict["Resources"]); r != nil {
		res = r
	}
	tm, tlm := in.tm, in.tlm
	in.depth++
	in.run(data, res)
	in.depth--
	in.gs, in.stack = saved, in.stack[:min(stack, len(in.stack))]
	in.tm, in.tlm = tm, tlm
}

func (in *interp) font(o any) *font {
	key := o
	if r, ok := o.(Ref); ok {
		key = r
	} else if _, ok := o.(Dict); ok {
		return in.p.f.loadFont(o) // an inline font dictionary: not cached
	}
	if ft, ok := in.fonts[key]; ok {
		return ft
	}
	ft := in.p.f.loadFont(o)
	in.fonts[key] = ft
	return ft
}

func (in *interp) nextLine() {
	in.tlm = matrix{1, 0, 0, 1, 0, -in.gs.leading}.mul(in.tlm)
	in.tm = in.tlm
}

func (in *interp) adjust(v float64) {
	tx := -v / 1000 * in.gs.size * in.gs.scale
	in.tm = matrix{1, 0, 0, 1, tx, 0}.mul(in.tm)
}

func (in *interp) show(s String) {
	ft := in.gs.font
	if ft == nil {
		return
	}
	gs := &in.gs
	invisible := gs.render == 3 || gs.render == 7
	for _, g := range ft.decode([]byte(s)) {
		trm := matrix{gs.size * gs.scale, 0, 0, gs.size, 0, gs.rise}.mul(in.tm).mul(gs.ctm)
		w := g.width
		bw := w
		if bw <= 0 {
			bw = 0.5
		}
		var xs, ys [4]float64
		for i, c := range [4][2]float64{{0, ft.descent}, {bw, ft.descent}, {0, ft.ascent}, {bw, ft.ascent}} {
			ux, uy := trm.apply(c[0], c[1])
			xs[i], ys[i] = in.p.toView(ux, uy)
		}
		ox, oy := in.p.toView(trm.apply(0, 0))
		ex, ey := in.p.toView(trm.apply(1, 0))
		ux, uy := in.p.toView(trm.apply(0, 1))
		dx, dy := ex-ox, ey-oy
		if n := math.Hypot(dx, dy); n > 0 {
			dx, dy = dx/n, dy/n
		}
		text := g.text
		gl := Glyph{
			Text: text, Known: g.known, Invisible: invisible,
			X0: min(xs[0], xs[1], xs[2], xs[3]), X1: max(xs[0], xs[1], xs[2], xs[3]),
			Y0: min(ys[0], ys[1], ys[2], ys[3]), Y1: max(ys[0], ys[1], ys[2], ys[3]),
			Dx: dx, Dy: dy, Size: math.Hypot(ux-ox, uy-oy),
			Space: strings.TrimSpace(text) == "" && text != "",
			Font:  ft.name, Name: g.name, OX: ox, OY: oy,
			Advance: (w*gs.size + gs.charSp) * gs.scale * math.Hypot(ex-ox, ey-oy) / max(gs.size*gs.scale, 1e-9),
		}
		if g.space && text == "" {
			gl.Space, gl.Known, gl.Text = true, true, " "
		}
		in.glyphs = append(in.glyphs, gl)
		tx := w*gs.size + gs.charSp
		if g.space {
			tx += gs.wordSp
		}
		in.tm = matrix{1, 0, 0, 1, tx * gs.scale, 0}.mul(in.tm)
	}
}

// replace turns the glyphs of a marked content with an /ActualText into one
// glyph holding that text.
func (in *interp) replace(m mark) {
	if m.start >= len(in.glyphs) {
		return
	}
	gs := in.glyphs[m.start:]
	g := gs[0]
	for _, o := range gs[1:] {
		g.X0, g.Y0 = min(g.X0, o.X0), min(g.Y0, o.Y0)
		g.X1, g.Y1 = max(g.X1, o.X1), max(g.Y1, o.Y1)
	}
	g.Text, g.Known = m.actual, true
	g.Space = strings.TrimSpace(m.actual) == "" && m.actual != ""
	in.glyphs = append(in.glyphs[:m.start], g)
}

func num(args []any, i int) float64 {
	if i >= len(args) {
		return 0
	}
	switch v := args[i].(type) {
	case int64:
		return float64(v)
	case float64:
		return v
	}
	return 0
}

func nums6(args []any) (matrix, bool) {
	if len(args) < 6 {
		return matrix{}, false
	}
	var m matrix
	for i := range 6 {
		switch v := args[len(args)-6+i].(type) {
		case int64:
			m[i] = float64(v)
		case float64:
			m[i] = v
		default:
			return m, false
		}
	}
	return m, true
}

// textString decodes a text string: UTF-16BE or UTF-8 with a byte order
// mark, else PDFDocEncoding.
func textString(b []byte) string {
	if len(b) >= 2 && b[0] == 0xFE && b[1] == 0xFF {
		return utf16Text(b[2 : 2+(len(b)-2)/2*2])
	}
	if len(b) >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF {
		return string(b[3:])
	}
	var sb strings.Builder
	for _, c := range b {
		if r, ok := pdfDoc[c]; ok {
			sb.WriteRune(r)
		} else {
			sb.WriteRune(rune(c))
		}
	}
	return sb.String()
}

var pdfDoc = map[byte]rune{
	0x18: '˘', 0x19: 'ˇ', 0x1A: 'ˆ', 0x1B: '˙', 0x1C: '˝', 0x1D: '˛', 0x1E: '˚', 0x1F: '˜',
	0x80: '•', 0x81: '†', 0x82: '‡', 0x83: '…', 0x84: '—', 0x85: '–', 0x86: 'ƒ', 0x87: '⁄',
	0x88: '‹', 0x89: '›', 0x8A: '−', 0x8B: '‰', 0x8C: '„', 0x8D: '“', 0x8E: '”', 0x8F: '‘',
	0x90: '’', 0x91: '‚', 0x92: '™', 0x93: 'ﬁ', 0x94: 'ﬂ', 0x95: 'Ł', 0x96: 'Œ', 0x97: 'Š',
	0x98: 'Ÿ', 0x99: 'Ž', 0x9A: 'ı', 0x9B: 'ł', 0x9C: 'œ', 0x9D: 'š', 0x9E: 'ž', 0xA0: '€',
}
