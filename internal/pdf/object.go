// Package pdf reads the text layer of PDF files: where each character of a
// page is drawn and which text it stands for. It renders nothing: images,
// paths and colours are skipped. It reads damaged files as far as it can (an
// xref table is rebuilt from the objects when it does not hold) and files
// encrypted with an empty user password, the usual case of PDFs that open
// without asking for one.
package pdf

import (
	"bytes"
	"fmt"
	"strconv"
)

// The objects of a PDF file: nil, bool, int64, float64, String, Name,
// Array, Dict, Ref and *Stream.
type (
	Name   string
	String string // the raw bytes of a string
	Array  []any
	Dict   map[Name]any
	Ref    struct{ Num, Gen int }
)

// A Stream is a dictionary and its data, still encoded.
type Stream struct {
	Dict Dict
	raw  []byte
	ref  Ref
}

type keyword string

// lexer reads the objects of a file or of a content stream.
type lexer struct {
	b   []byte
	pos int
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == 0
}

func isDelim(c byte) bool {
	switch c {
	case '(', ')', '<', '>', '[', ']', '{', '}', '/', '%':
		return true
	}
	return false
}

func (l *lexer) skipSpace() {
	for l.pos < len(l.b) {
		c := l.b[l.pos]
		if c == '%' {
			for l.pos < len(l.b) && l.b[l.pos] != '\n' && l.b[l.pos] != '\r' {
				l.pos++
			}
			continue
		}
		if !isSpace(c) {
			return
		}
		l.pos++
	}
}

// token reads one token: a number, a Name, a String, a keyword (true,
// false, null, obj, R, an operator) or one of the delimiters "[", "]", "<<",
// ">>", "{", "}" as a keyword. It returns nil at the end of the data.
func (l *lexer) token() any {
	l.skipSpace()
	if l.pos >= len(l.b) {
		return nil
	}
	c := l.b[l.pos]
	switch c {
	case '[', ']', '{', '}':
		l.pos++
		return keyword(c)
	case '<':
		if l.pos+1 < len(l.b) && l.b[l.pos+1] == '<' {
			l.pos += 2
			return keyword("<<")
		}
		return l.hexString()
	case '>':
		if l.pos+1 < len(l.b) && l.b[l.pos+1] == '>' {
			l.pos += 2
			return keyword(">>")
		}
		l.pos++
		return keyword(">")
	case '(':
		return l.literalString()
	case ')':
		l.pos++
		return keyword(")")
	case '/':
		l.pos++
		start := l.pos
		for l.pos < len(l.b) && !isSpace(l.b[l.pos]) && !isDelim(l.b[l.pos]) {
			l.pos++
		}
		return Name(unescapeName(l.b[start:l.pos]))
	}
	start := l.pos
	for l.pos < len(l.b) && !isSpace(l.b[l.pos]) && !isDelim(l.b[l.pos]) {
		l.pos++
	}
	if l.pos == start { // a stray delimiter
		l.pos++
		return keyword(l.b[start:l.pos])
	}
	w := l.b[start:l.pos]
	if n, ok := number(w); ok {
		return n
	}
	switch string(w) {
	case "true":
		return true
	case "false":
		return false
	case "null":
		return nil
	}
	return keyword(w)
}

func number(w []byte) (any, bool) {
	c := w[0]
	if !(c >= '0' && c <= '9' || c == '-' || c == '+' || c == '.') {
		return nil, false
	}
	if i, err := strconv.ParseInt(string(w), 10, 64); err == nil {
		return i, true
	}
	if f, err := strconv.ParseFloat(string(w), 64); err == nil {
		return f, true
	}
	// Producers write "--5", "5-" or "0.0.1": read what precedes the junk.
	s := string(bytes.TrimLeft(w, "+-"))
	neg := bytes.Count(w[:len(w)-len(s)], []byte("-"))%2 == 1
	end := 0
	dot := false
	for end < len(s) && (s[end] >= '0' && s[end] <= '9' || s[end] == '.' && !dot) {
		dot = dot || s[end] == '.'
		end++
	}
	if end == 0 {
		return nil, false
	}
	f, err := strconv.ParseFloat(s[:end], 64)
	if err != nil {
		return nil, false
	}
	if neg {
		f = -f
	}
	return f, true
}

func unescapeName(b []byte) string {
	if bytes.IndexByte(b, '#') < 0 {
		return string(b)
	}
	out := make([]byte, 0, len(b))
	for i := 0; i < len(b); i++ {
		if b[i] == '#' && i+2 < len(b) {
			if v, err := strconv.ParseUint(string(b[i+1:i+3]), 16, 8); err == nil {
				out = append(out, byte(v))
				i += 2
				continue
			}
		}
		out = append(out, b[i])
	}
	return string(out)
}

func unhex(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

func (l *lexer) hexString() String {
	l.pos++ // <
	var out []byte
	var hi byte
	half := false
	for l.pos < len(l.b) && l.b[l.pos] != '>' {
		v, ok := unhex(l.b[l.pos])
		l.pos++
		if !ok {
			continue
		}
		if half {
			out = append(out, hi<<4|v)
		} else {
			hi = v
		}
		half = !half
	}
	if half {
		out = append(out, hi<<4)
	}
	l.pos++ // >
	return String(out)
}

func (l *lexer) literalString() String {
	l.pos++ // (
	var out []byte
	depth := 1
	for l.pos < len(l.b) {
		c := l.b[l.pos]
		l.pos++
		switch c {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return String(out)
			}
		case '\\':
			if l.pos >= len(l.b) {
				return String(out)
			}
			e := l.b[l.pos]
			l.pos++
			switch e {
			case 'n':
				c = '\n'
			case 'r':
				c = '\r'
			case 't':
				c = '\t'
			case 'b':
				c = '\b'
			case 'f':
				c = '\f'
			case '\r':
				if l.pos < len(l.b) && l.b[l.pos] == '\n' {
					l.pos++
				}
				continue
			case '\n':
				continue
			default:
				if e >= '0' && e <= '7' {
					v := int(e - '0')
					for k := 0; k < 2 && l.pos < len(l.b) && l.b[l.pos] >= '0' && l.b[l.pos] <= '7'; k++ {
						v = v*8 + int(l.b[l.pos]-'0')
						l.pos++
					}
					c = byte(v)
				} else {
					c = e
				}
			}
		case '\r': // an end of line in a string is a \n
			if l.pos < len(l.b) && l.b[l.pos] == '\n' {
				l.pos++
			}
			c = '\n'
		}
		out = append(out, c)
	}
	return String(out)
}

// object reads one object, with arrays, dictionaries and references
// ("1 0 R") built. Keywords other than those come back as keyword.
func (l *lexer) object(depth int) any {
	t := l.token()
	return l.build(t, depth)
}

func (l *lexer) build(t any, depth int) any {
	if depth > 100 {
		return nil
	}
	switch v := t.(type) {
	case keyword:
		switch v {
		case "[":
			var a Array
			for {
				save := l.pos
				t := l.token()
				if t == nil {
					return a
				}
				if k, ok := t.(keyword); ok && (k == "]" || k == ">>" || k == "endobj" || k == "stream") {
					if k != "]" {
						l.pos = save
					}
					return a
				}
				a = append(a, l.build(t, depth+1))
			}
		case "<<":
			d := Dict{}
			for {
				save := l.pos
				t := l.token()
				if t == nil {
					return d
				}
				if k, ok := t.(keyword); ok {
					if k == ">>" {
						return d
					}
					if k == "endobj" || k == "stream" || k == "]" {
						l.pos = save
						return d
					}
					continue
				}
				key, ok := t.(Name)
				if !ok {
					continue
				}
				save = l.pos
				vt := l.token()
				if k, ok := vt.(keyword); ok && (k == ">>" || k == "endobj") {
					l.pos = save
					continue
				}
				d[key] = l.build(vt, depth+1)
			}
		}
		return v
	case int64:
		// "num gen R"
		save := l.pos
		if g, ok := l.token().(int64); ok {
			if k, ok := l.token().(keyword); ok && k == "R" {
				return Ref{int(v), int(g)}
			}
		}
		l.pos = save
		return v
	}
	return t
}

func (r Ref) String() string { return fmt.Sprintf("%d %d R", r.Num, r.Gen) }
