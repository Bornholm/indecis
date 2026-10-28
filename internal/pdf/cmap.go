package pdf

import (
	"strings"
	"unicode/utf16"
)

// A cmap maps character codes, one to four bytes long, to text (a
// ToUnicode map) or to CIDs (a font's encoding).
type cmap struct {
	space  []codeRange // codespace ranges
	chars  map[codeKey]string
	cids   map[codeKey]int
	ranges []mapRange
}

type codeKey struct {
	n    int // bytes
	code uint32
}

type codeRange struct {
	n      int
	lo, hi []byte
}

type mapRange struct {
	n      int
	lo, hi uint32
	text   string   // bfrange: the text of lo; the last code unit counts up
	texts  []string // bfrange with an array: one text per code
	cid    int      // cidrange: the CID of lo
	isCID  bool
}

func parseCMap(data []byte) *cmap {
	m := &cmap{chars: map[codeKey]string{}, cids: map[codeKey]int{}}
	l := &lexer{b: data}
	var stack []any
	for {
		t := l.token()
		if t == nil {
			break
		}
		k, ok := t.(keyword)
		if !ok {
			stack = append(stack, l.build(t, 0))
			continue
		}
		switch k {
		case "[", "<<":
			stack = append(stack, l.build(t, 0))
			continue
		case "endcodespacerange":
			for i := 0; i+1 < len(stack); i += 2 {
				lo, ok1 := stack[i].(String)
				hi, ok2 := stack[i+1].(String)
				if ok1 && ok2 && len(lo) == len(hi) && len(lo) > 0 && len(lo) <= 4 {
					m.space = append(m.space, codeRange{len(lo), []byte(lo), []byte(hi)})
				}
			}
		case "endbfchar":
			for i := 0; i+1 < len(stack); i += 2 {
				src, ok := stack[i].(String)
				if !ok || len(src) == 0 || len(src) > 4 {
					continue
				}
				switch dst := stack[i+1].(type) {
				case String:
					m.chars[codeKey{len(src), be(src)}] = utf16Text([]byte(dst))
				case Name:
					m.chars[codeKey{len(src), be(src)}] = glyphText(string(dst))
				}
			}
		case "endbfrange":
			for i := 0; i+2 < len(stack); i += 3 {
				lo, ok1 := stack[i].(String)
				hi, ok2 := stack[i+1].(String)
				if !ok1 || !ok2 || len(lo) != len(hi) || len(lo) == 0 || len(lo) > 4 {
					continue
				}
				r := mapRange{n: len(lo), lo: be(lo), hi: be(hi)}
				if r.hi < r.lo {
					continue
				}
				switch dst := stack[i+2].(type) {
				case String:
					r.text = utf16Text([]byte(dst))
				case Array:
					for _, d := range dst {
						s, _ := d.(String)
						r.texts = append(r.texts, utf16Text([]byte(s)))
					}
				default:
					continue
				}
				m.ranges = append(m.ranges, r)
			}
		case "endcidchar":
			for i := 0; i+1 < len(stack); i += 2 {
				src, ok1 := stack[i].(String)
				cid, ok2 := stack[i+1].(int64)
				if ok1 && ok2 && len(src) > 0 && len(src) <= 4 {
					m.cids[codeKey{len(src), be(src)}] = int(cid)
				}
			}
		case "endcidrange":
			for i := 0; i+2 < len(stack); i += 3 {
				lo, ok1 := stack[i].(String)
				hi, ok2 := stack[i+1].(String)
				cid, ok3 := stack[i+2].(int64)
				if ok1 && ok2 && ok3 && len(lo) == len(hi) && len(lo) > 0 && len(lo) <= 4 {
					m.ranges = append(m.ranges, mapRange{n: len(lo), lo: be(lo), hi: be(hi), cid: int(cid), isCID: true})
				}
			}
		}
		stack = stack[:0]
	}
	return m
}

func be(s String) uint32 {
	var v uint32
	for i := 0; i < len(s); i++ {
		v = v<<8 | uint32(s[i])
	}
	return v
}

// utf16Text reads a ToUnicode destination: UTF-16BE, or single bytes when
// its length is odd (a frequent mistake of producers).
func utf16Text(b []byte) string {
	if len(b)%2 == 1 {
		var sb strings.Builder
		for _, c := range b {
			sb.WriteRune(rune(c))
		}
		return sb.String()
	}
	u := make([]uint16, len(b)/2)
	for i := range u {
		u[i] = uint16(b[2*i])<<8 | uint16(b[2*i+1])
	}
	return string(utf16.Decode(u))
}

// next splits the code that starts s: its length from the codespace
// ranges, or def bytes when none matches.
func (m *cmap) next(s []byte, def int) int {
	if m != nil && len(m.space) > 0 {
		shortest := 4
		for n := 1; n <= 4 && n <= len(s); n++ {
			for _, r := range m.space {
				if r.n != n {
					continue
				}
				shortest = min(shortest, r.n)
				ok := true
				for i := range n {
					if s[i] < r.lo[i] || s[i] > r.hi[i] {
						ok = false
						break
					}
				}
				if ok {
					return n
				}
			}
		}
		for _, r := range m.space {
			shortest = min(shortest, r.n)
		}
		return max(1, min(shortest, len(s)))
	}
	return max(1, min(def, len(s)))
}

func (m *cmap) text(n int, code uint32) (string, bool) {
	if m == nil {
		return "", false
	}
	if t, ok := m.chars[codeKey{n, code}]; ok {
		return t, true
	}
	for _, r := range m.ranges {
		if r.isCID || r.n != n || code < r.lo || code > r.hi {
			continue
		}
		off := code - r.lo
		if r.texts != nil {
			if int(off) < len(r.texts) {
				return r.texts[off], true
			}
			continue
		}
		rs := []rune(r.text)
		if len(rs) == 0 {
			return "", true
		}
		rs[len(rs)-1] += rune(off)
		return string(rs), true
	}
	return "", false
}

func (m *cmap) cid(n int, code uint32) (int, bool) {
	if m == nil {
		return 0, false
	}
	if c, ok := m.cids[codeKey{n, code}]; ok {
		return c, true
	}
	for _, r := range m.ranges {
		if r.isCID && r.n == n && code >= r.lo && code <= r.hi {
			return r.cid + int(code-r.lo), true
		}
	}
	return 0, false
}
