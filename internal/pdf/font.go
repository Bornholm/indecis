package pdf

import (
	"bytes"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode/utf16"
)

// A font turns the bytes of a string shown with it into codes, the text
// of each code and its width.
type font struct {
	typ       Name // Type1, MMType1, TrueType, Type3 or Type0
	twoByte   bool // Type0: codes are split by encoding (Identity: 2 bytes)
	encoding  *cmap
	ucs2      bool // Type0 with a predefined Uni…-UCS2/UTF16 encoding: codes are text
	toUnicode *cmap
	text      [256]string // simple fonts: text by code, from the encoding
	names     [256]string // simple fonts: glyph name by code
	widths    map[int]float64
	defWidth  float64
	ascent    float64 // em fractions
	descent   float64
	scale     float64 // glyph units to ems (0.001, or a Type3 font matrix)
	gidText   map[int]string
	name      string // the base font's name, its subset tag left out
}

type coreFont struct {
	ascent, descent int
	widths          string
}

var (
	coreOnce    sync.Once
	coreWidths  map[string]map[string]float64
	glyphOnce   sync.Once
	glyphTable  map[string]string
	subsetName  = regexp.MustCompile(`^[A-Z]{6}\+`)
	uniName     = regexp.MustCompile(`^uni([0-9A-Fa-f]{4})+$`)
	uName       = regexp.MustCompile(`^u([0-9A-Fa-f]{4,6})$`)
	numericName = regexp.MustCompile(`^(?:g|glyph|cid|index|c|G|C|#)?(\d+)$`)
)

// glyphText is the text of a glyph name: the Adobe Glyph List, uniXXXX,
// uXXXX[XX], ligature names (f_f_i) and suffixes (a.sc). "" when unknown.
func glyphText(name string) string {
	glyphOnce.Do(func() {
		glyphTable = map[string]string{}
		for _, l := range strings.Split(glyphList, "\n") {
			if n, t, ok := strings.Cut(l, " "); ok {
				glyphTable[n] = t
			}
		}
	})
	if t, ok := glyphTable[name]; ok {
		if r := []rune(t); len(r) == 1 && r[0] >= 0xE000 && r[0] <= 0xF8FF {
			// Adobe's private-use glyphs: small capitals are letters, the
			// rest (pieces of big delimiters) stand for no text.
			if b, ok := strings.CutSuffix(name, "small"); ok && len(b) == 1 {
				return strings.ToLower(b)
			}
			return ""
		}
		return t
	}
	if t, ok := texGlyphs[name]; ok {
		return t
	}
	for _, suf := range texSizes {
		if b, ok := strings.CutSuffix(name, suf); ok && b != "" {
			return glyphText(b)
		}
	}
	if i := strings.IndexByte(name, '.'); i > 0 {
		return glyphText(name[:i])
	}
	if strings.Contains(name, "_") {
		var sb strings.Builder
		for _, part := range strings.Split(name, "_") {
			t := glyphText(part)
			if t == "" {
				return ""
			}
			sb.WriteString(t)
		}
		return sb.String()
	}
	if uniName.MatchString(name) {
		var u []uint16
		for i := 3; i+4 <= len(name); i += 4 {
			v, _ := strconv.ParseUint(name[i:i+4], 16, 16)
			u = append(u, uint16(v))
		}
		return string(utf16.Decode(u))
	}
	if m := uName.FindStringSubmatch(name); m != nil {
		v, _ := strconv.ParseUint(m[1], 16, 32)
		if v <= 0x10FFFF && !(v >= 0xD800 && v <= 0xDFFF) {
			return string(rune(v))
		}
	}
	return ""
}

// texGlyphs holds the names of TeX's math fonts that the glyph list lacks;
// "" for pieces of big delimiters and accents that stand for no text.
var texGlyphs = map[string]string{
	"bardbl": "‖", "greaterorequalslant": "⩾", "lessorequalslant": "⩽", "negationslash": "",
	"mapsto": "↦", "prime": "′", "square": "□", "bigcircle": "◯", "nequal": "≠",
	"vextendsingle": "", "vextenddouble": "", "visualspace": "␣",
	"angbracketleft": "⟨", "angbracketright": "⟩", "floorleft": "⌊", "floorright": "⌋",
	"ceilingleft": "⌈", "ceilingright": "⌉", "hat": "^", "tilde": "~", "check": "ˇ",
	"bracehtipdownleft": "", "bracehtipdownright": "", "bracehtipupleft": "", "bracehtipupright": "",
	"slashbig": "/", "backslashbig": "\\", "follows": "≻", "precedes": "≺", "turnstileleft": "⊢",
	"turnstileright": "⊣", "latticetop": "⊤", "owner": "∋", "unionsq": "⊔", "intersectionsq": "⊓",
	"unionmulti": "⊎", "contintegral": "∮", "coproduct": "∐", "triangleleft": "◁", "triangleright": "▷",
	"star": "⋆", "dotlessj": "ȷ", "ff": "ff", "ffi": "ffi", "ffl": "ffl", "minusplus": "∓",
	"arrowdblleft": "⇐", "arrowdblright": "⇒", "arrowdblboth": "⇔", "arrownortheast": "↗",
	"arrowsoutheast": "↘", "arrownorthwest": "↖", "arrowsouthwest": "↙", "similarequal": "≃",
	"lessmuch": "≪", "greatermuch": "≫", "equivasymptotic": "≍", "approxequal": "≈",
	"reflexsubset": "⊆", "reflexsuperset": "⊇", "element": "∈", "infinity": "∞",
	// The AMS fonts (msam, msbm), stmaryrd, txfonts.
	"lessorsimilar": "≲", "greaterorsimilar": "≳", "lessorapproxeql": "⪅", "greaterorapproxeql": "⪆",
	"planckover2pi": "ℏ", "planckover2pi1": "ℏ", "forces": "⊩", "defines": "≜", "triangleleftequal": "⊴",
	"trianglerightequal": "⊵", "subsetdblequal": "⫅", "supersetdblequal": "⫆", "dotplus": "∔",
	"multimap": "⊸", "intercal": "⊺", "between": "≬", "pitchfork": "⋔", "squaresolid": "■",
	"lozenge": "◊", "lozengesolid": "⧫", "llbracket": "⟦", "rrbracket": "⟧", "colonequal": "≔",
	"shortrightarrow": "→", "shortleftarrow": "←", "arrowvert": "", "dotacc": "˙",
}

// texSizes are the suffixes of the sizes of TeX's big operators and
// delimiters (summationdisplay, parenleftBig).
var texSizes = []string{"display", "text", "bigg", "Bigg", "big", "Big", "wide", "wider", "widest", "single", "double"}

// coreName is the standard font a font name stands for, if any: Arial is
// drawn with Helvetica's widths, Times New Roman with Times'.
func coreName(base string) string {
	base = subsetName.ReplaceAllString(base, "")
	if _, ok := coreFonts[base]; ok {
		return base
	}
	l := strings.ToLower(base)
	bold := strings.Contains(l, "bold") || strings.Contains(l, "black") || strings.Contains(l, "heavy")
	italic := strings.Contains(l, "italic") || strings.Contains(l, "oblique")
	switch {
	case strings.Contains(l, "symbol"):
		return "Symbol"
	case strings.Contains(l, "dingbat"):
		return "ZapfDingbats"
	case strings.Contains(l, "courier") || strings.Contains(l, "mono"):
		return "Courier" + suffix(bold, italic, "Oblique")
	case strings.Contains(l, "times") || strings.Contains(l, "roman") || strings.Contains(l, "serif") && !strings.Contains(l, "sans"):
		if !bold && !italic {
			return "Times-Roman"
		}
		return "Times" + suffix(bold, italic, "Italic")
	}
	return "Helvetica" + suffix(bold, italic, "Oblique")
}

func suffix(bold, italic bool, slant string) string {
	switch {
	case bold && italic:
		return "-Bold" + slant
	case bold:
		return "-Bold"
	case italic:
		return "-" + slant
	}
	return ""
}

func coreWidth(font, glyph string) (float64, bool) {
	coreOnce.Do(func() {
		coreWidths = map[string]map[string]float64{}
		for name, c := range coreFonts {
			m := map[string]float64{}
			f := strings.Fields(c.widths)
			for i := 0; i+1 < len(f); i += 2 {
				w, _ := strconv.ParseFloat(f[i+1], 64)
				m[f[i]] = w
			}
			coreWidths[name] = m
		}
	})
	w, ok := coreWidths[font][glyph]
	return w, ok
}

func (f *File) loadFont(o any) *font {
	d := f.dict(o)
	if d == nil {
		return nil
	}
	ft := &font{typ: f.name(d["Subtype"]), widths: map[int]float64{}, scale: 0.001}
	if s, ok := f.Resolve(d["ToUnicode"]).(*Stream); ok {
		if data, err := f.decode(s); err == nil || len(data) > 0 {
			ft.toUnicode = parseCMap(data)
		}
	}
	base := string(f.name(d["BaseFont"]))
	ft.name = subsetName.ReplaceAllString(base, "")
	desc := f.dict(d["FontDescriptor"])
	if ft.typ == "Type0" {
		ft.twoByte = true
		switch e := f.Resolve(d["Encoding"]).(type) {
		case Name:
			s := string(e)
			if strings.HasPrefix(s, "Uni") && (strings.Contains(s, "UCS2") || strings.Contains(s, "UTF16")) {
				ft.ucs2 = true
			}
		case *Stream:
			if data, err := f.decode(e); err == nil || len(data) > 0 {
				ft.encoding = parseCMap(data)
			}
		}
		var cid Dict
		if kids := f.array(d["DescendantFonts"]); len(kids) > 0 {
			cid = f.dict(kids[0])
		}
		ft.defWidth = 1000
		if cid != nil {
			if v, ok := f.num(cid["DW"]); ok {
				ft.defWidth = v
			}
			w := f.array(cid["W"])
			for i := 0; i < len(w); {
				first, ok := f.num(w[i])
				if !ok || i+1 >= len(w) {
					break
				}
				if a, ok := f.Resolve(w[i+1]).(Array); ok {
					for k, v := range a {
						if x, ok := f.num(v); ok {
							ft.widths[int(first)+k] = x
						}
					}
					i += 2
					continue
				}
				last, ok1 := f.num(w[i+1])
				if i+2 >= len(w) || !ok1 {
					break
				}
				x, _ := f.num(w[i+2])
				for c := int(first); c <= int(last) && c-int(first) < 65536; c++ {
					ft.widths[c] = x
				}
				i += 3
			}
			desc = f.dict(cid["FontDescriptor"])
			if ft.toUnicode == nil && !ft.ucs2 && desc != nil {
				// No map to text: the glyphs of an embedded TrueType font may
				// say which characters they draw.
				if s, ok := f.Resolve(desc["FontFile2"]).(*Stream); ok {
					if data, err := f.decode(s); err == nil {
						ft.gidText = trueTypeText(data)
					}
				}
			}
		}
		ft.metrics(f, desc, base)
		return ft
	}
	// Simple fonts.
	if ft.typ == "Type3" {
		if m := f.array(d["FontMatrix"]); len(m) == 6 {
			if v, ok := f.num(m[0]); ok && v != 0 {
				ft.scale = v
			}
		}
	}
	core := coreName(base)
	first, _ := f.num(d["FirstChar"])
	widths := f.array(d["Widths"])
	for i, w := range widths {
		if x, ok := f.num(w); ok {
			ft.widths[int(first)+i] = x
		}
	}
	if desc != nil {
		if v, ok := f.num(desc["MissingWidth"]); ok {
			ft.defWidth = v
		}
	}
	// The encoding: a base, changed by Differences.
	var names [256]string
	symbolic := false
	if desc != nil {
		if fl, ok := f.num(desc["Flags"]); ok {
			symbolic = int(fl)&4 != 0
		}
	}
	baseEnc := standardEncoding
	switch {
	case core == "Symbol" && (desc == nil || desc["FontFile"] == nil && desc["FontFile2"] == nil && desc["FontFile3"] == nil) || strings.Contains(strings.ToLower(base), "symbol"):
		baseEnc = symbolEncoding
	case core == "ZapfDingbats":
		baseEnc = zapfEncoding
	case ft.typ == "TrueType":
		baseEnc = winAnsiEncoding
	case desc != nil:
		if s, ok := f.Resolve(desc["FontFile"]).(*Stream); ok {
			if data, err := f.decode(s); err == nil || len(data) > 0 {
				if enc, ok := type1Encoding(data); ok {
					baseEnc = enc
				}
			}
		}
	}
	_ = symbolic
	enc := f.Resolve(d["Encoding"])
	encDict, _ := enc.(Dict)
	if encDict != nil {
		enc = f.Resolve(encDict["BaseEncoding"])
	}
	switch enc {
	case Name("WinAnsiEncoding"):
		baseEnc = winAnsiEncoding
	case Name("MacRomanEncoding"):
		baseEnc = macRomanEncoding
	case Name("StandardEncoding"):
		baseEnc = standardEncoding
	case Name("MacExpertEncoding"):
	}
	names = baseEnc
	if encDict != nil {
		code := 0
		for _, v := range f.array(encDict["Differences"]) {
			switch x := f.Resolve(v).(type) {
			case int64:
				code = int(x)
			case float64:
				code = int(x)
			case Name:
				if code >= 0 && code < 256 {
					names[code] = string(x)
				}
				code++
			}
		}
	}
	ft.names = names
	for c, n := range names {
		if n == "" {
			continue
		}
		t := glyphText(n)
		if t == "" && (ft.typ == "Type3" || numericName.MatchString(n)) && c >= 32 && c < 127 {
			t = string(rune(c)) // a Type3 or numbered glyph: its code is the best guess
		}
		ft.text[c] = t
	}
	if strings.HasPrefix(strings.ToLower(ft.name), "esint") {
		// esint's fonts hold integral signs only, under the names of
		// another encoding.
		for c := range 256 {
			ft.text[c], ft.names[c] = "∫", "integraldisplay"
		}
		ft.toUnicode = nil
	}
	if table, ok := mathabxText[mathabxFont(ft.name)]; ok {
		// Copies of the mathabx fonts name their glyphs and map them to
		// text after another encoding: the slots say what they are.
		ft.toUnicode = nil
		for c := range 256 {
			// The slots of sizes of big symbols that the table lacks stand
			// for no text, as pieces of delimiters.
			ft.text[c], ft.names[c] = table[byte(c)], "mathabx"
		}
	}
	if len(widths) == 0 {
		for c, n := range names {
			if w, ok := coreWidth(core, n); ok {
				ft.widths[c] = w
			}
		}
		if ft.defWidth == 0 {
			ft.defWidth = 500
		}
	}
	ft.metrics(f, desc, core)
	if ft.typ == "Type3" {
		if b := f.rect(d["FontBBox"]); b != nil && b[3] > b[1] {
			m := f.array(d["FontMatrix"])
			sy := 0.001
			if len(m) == 6 {
				if v, ok := f.num(m[3]); ok && v != 0 {
					sy = v
				}
			}
			if a, de := b[3]*sy, b[1]*sy; a > 0.3 && a < 1.5 && de <= 0 && de > -0.6 {
				ft.ascent, ft.descent = a, de
			}
		}
	}
	return ft
}

// mathabxFont is the mathabx font (matha, mathb, mathx) a font's name
// stands for: TeX-matha10, matha7.
func mathabxFont(name string) string {
	n := strings.TrimRight(strings.TrimPrefix(strings.ToLower(name), "tex-"), "0123456789")
	switch n {
	case "matha", "mathb", "mathx":
		return n
	}
	return ""
}

// metrics sets the font's ascent and descent: its descriptor's, else the
// standard font's, else typical values.
func (ft *font) metrics(f *File, desc Dict, base string) {
	ft.ascent, ft.descent = 0.8, -0.2
	if c, ok := coreFonts[coreName(base)]; ok {
		ft.ascent, ft.descent = float64(c.ascent)/1000, float64(c.descent)/1000
	}
	if desc == nil {
		return
	}
	a, ok1 := f.num(desc["Ascent"])
	d, ok2 := f.num(desc["Descent"])
	if ok1 && ok2 && a > 300 && a < 1500 && d <= 0 && d > -600 {
		ft.ascent, ft.descent = a/1000, d/1000
	}
}

// A code shown with a font.
type glyphCode struct {
	text  string
	width float64 // ems
	space bool    // a one-byte code 32, which word spacing widens
	known bool    // the font says which text the code stands for
	name  string  // simple fonts: the glyph's name
}

func (ft *font) decode(s []byte) []glyphCode {
	var out []glyphCode
	for len(s) > 0 {
		var n int
		switch {
		case !ft.twoByte:
			n = ft.toUnicode.next(s, 1)
			if ft.toUnicode == nil || len(ft.toUnicode.space) == 0 || n != 1 {
				n = 1
			}
		case ft.encoding != nil:
			n = ft.encoding.next(s, 2)
		case ft.ucs2:
			n = min(2, len(s))
			if n == 2 && s[0] >= 0xD8 && s[0] <= 0xDB && len(s) >= 4 {
				n = 4
			}
		default:
			n = ft.toUnicode.next(s, 2)
			if n != 2 && len(s) >= 2 {
				n = 2
			}
		}
		code := uint32(0)
		for i := range n {
			code = code<<8 | uint32(s[i])
		}
		s = s[n:]
		g := glyphCode{space: n == 1 && code == 32}
		cid := int(code)
		if ft.encoding != nil {
			if c, ok := ft.encoding.cid(n, code); ok {
				cid = c
			}
		}
		if t, ok := ft.toUnicode.text(n, code); ok {
			g.text, g.known = t, true
		} else if ft.ucs2 {
			b := []byte{byte(code >> 24), byte(code >> 16), byte(code >> 8), byte(code)}
			g.text, g.known = utf16Text(b[4-n:]), true
		} else if ft.twoByte {
			g.text, g.known = ft.gidText[cid]
			if g.text == "" && ft.gidText == nil {
				g.known = false
			}
		} else if code < 256 {
			g.text = ft.text[code]
			// A named glyph with no text (a piece of a big delimiter) is
			// known to stand for nothing; a code with no name is not text.
			g.known = ft.names[code] != ""
		}
		if !ft.twoByte && code < 256 {
			g.name = ft.names[code]
		}
		w, ok := ft.widths[cid]
		if !ok {
			w = ft.defWidth
		}
		g.width = w * ft.scale
		out = append(out, g)
	}
	return out
}

// type1Encoding reads the built-in encoding of a Type 1 font program
// ("dup 65 /A put" in its clear-text part).
func type1Encoding(data []byte) ([256]string, bool) {
	var enc [256]string
	i := bytes.Index(data, []byte("/Encoding"))
	if i < 0 {
		return enc, false
	}
	rest := data[i:]
	if bytes.HasPrefix(bytes.TrimSpace(rest[len("/Encoding"):]), []byte("StandardEncoding")) {
		return standardEncoding, true
	}
	end := bytes.Index(rest, []byte("readonly def"))
	if end < 0 {
		end = min(len(rest), 20000)
	}
	l := &lexer{b: rest[:end]}
	found := false
	var prev []any
	for {
		t := l.token()
		if t == nil {
			break
		}
		if k, ok := t.(keyword); ok && k == "put" && len(prev) >= 2 {
			c, ok1 := prev[len(prev)-2].(int64)
			n, ok2 := prev[len(prev)-1].(Name)
			if ok1 && ok2 && c >= 0 && c < 256 {
				enc[c] = string(n)
				found = true
			}
		}
		prev = append(prev, t)
		if len(prev) > 4 {
			prev = prev[1:]
		}
	}
	return enc, found
}
