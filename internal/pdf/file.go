package pdf

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"strconv"
)

type xrefEntry struct {
	off    int64 // byte offset, or the object stream's number
	stream bool  // in an object stream, at index off2
	idx    int
}

// A File is an open PDF file.
type File struct {
	data    []byte
	xref    map[int]xrefEntry
	trailer Dict
	cache   map[int]any
	loading map[int]bool
	crypt   *crypt
}

// Open reads the structure of a PDF file held in data.
func Open(data []byte) (*File, error) {
	f := &File{data: data, xref: map[int]xrefEntry{}, cache: map[int]any{}, loading: map[int]bool{}}
	if err := f.readXref(); err != nil || f.dict(f.trailer["Root"]) == nil {
		// A broken or missing xref table: find the objects themselves.
		f.xref, f.trailer, f.cache = map[int]xrefEntry{}, nil, map[int]any{}
		if err := f.rebuildXref(); err != nil {
			return nil, err
		}
	}
	if enc, ok := f.trailer["Encrypt"]; ok && enc != nil {
		c, err := newCrypt(f, enc)
		if err != nil {
			return nil, err
		}
		f.crypt = c
		f.cache = map[int]any{}
	}
	if f.dict(f.trailer["Root"]) == nil {
		return nil, errors.New("pdf: no document catalog")
	}
	return f, nil
}

func (f *File) readXref() error {
	tail := f.data[max(0, len(f.data)-2048):]
	i := bytes.LastIndex(tail, []byte("startxref"))
	if i < 0 {
		return errors.New("pdf: no startxref")
	}
	l := &lexer{b: tail, pos: i + len("startxref")}
	off, ok := l.token().(int64)
	if !ok {
		return errors.New("pdf: bad startxref")
	}
	seen := map[int64]bool{}
	for off > 0 && !seen[off] && off < int64(len(f.data)) {
		seen[off] = true
		trailer, err := f.readSection(off)
		if err != nil {
			return err
		}
		if f.trailer == nil {
			f.trailer = trailer
		}
		// A hybrid file keeps part of its xref in a stream.
		if x, ok := trailer["XRefStm"].(int64); ok && !seen[x] {
			seen[x] = true
			if _, err := f.readSection(x); err != nil {
				return err
			}
		}
		prev, _ := trailer["Prev"].(int64)
		off = prev
	}
	if f.trailer == nil {
		return errors.New("pdf: no trailer")
	}
	return nil
}

// readSection reads an xref table or an xref stream at off, adding the
// entries not already known (later sections come first).
func (f *File) readSection(off int64) (Dict, error) {
	l := &lexer{b: f.data, pos: int(off)}
	t := l.token()
	if k, ok := t.(keyword); ok && k == "xref" {
		for {
			save := l.pos
			first, ok := l.token().(int64)
			if !ok {
				l.pos = save
				break
			}
			n, ok := l.token().(int64)
			if !ok {
				return nil, errors.New("pdf: bad xref subsection")
			}
			for k := int64(0); k < n; k++ {
				o, _ := l.token().(int64)
				g, _ := l.token().(int64)
				typ, _ := l.token().(keyword)
				num := int(first + k)
				if _, known := f.xref[num]; known {
					continue
				}
				if typ == "n" && o > 0 {
					f.xref[num] = xrefEntry{off: o}
				} else if typ == "f" {
					f.xref[num] = xrefEntry{off: -1}
				}
				_ = g
			}
		}
		if k, ok := l.token().(keyword); !ok || k != "trailer" {
			return nil, errors.New("pdf: no trailer after xref")
		}
		d, ok := l.object(0).(Dict)
		if !ok {
			return nil, errors.New("pdf: bad trailer")
		}
		return d, nil
	}
	// An xref stream: "n g obj << ... >> stream".
	l.pos = int(off)
	_, obj, err := f.objectAt(int(off), false)
	if err != nil {
		return nil, err
	}
	s, ok := obj.(*Stream)
	if !ok || s.Dict["Type"] != Name("XRef") {
		return nil, errors.New("pdf: no xref at startxref")
	}
	data, err := f.decode(s)
	if err != nil {
		return nil, err
	}
	w := f.array(s.Dict["W"])
	if len(w) < 3 {
		return nil, errors.New("pdf: bad xref stream widths")
	}
	ws := [3]int{}
	for i := range 3 {
		v, _ := w[i].(int64)
		ws[i] = int(v)
	}
	index := f.array(s.Dict["Index"])
	if index == nil {
		size, _ := s.Dict["Size"].(int64)
		index = Array{int64(0), size}
	}
	row := ws[0] + ws[1] + ws[2]
	p := 0
	field := func(n int, def int64) int64 {
		if n == 0 {
			return def
		}
		var v int64
		for range n {
			v = v<<8 | int64(data[p])
			p++
		}
		return v
	}
	for i := 0; i+1 < len(index); i += 2 {
		first, _ := index[i].(int64)
		n, _ := index[i+1].(int64)
		for k := int64(0); k < n && p+row <= len(data); k++ {
			typ := field(ws[0], 1)
			a := field(ws[1], 0)
			b := field(ws[2], 0)
			num := int(first + k)
			if _, known := f.xref[num]; known {
				continue
			}
			switch typ {
			case 0:
				f.xref[num] = xrefEntry{off: -1}
			case 1:
				f.xref[num] = xrefEntry{off: a}
			case 2:
				f.xref[num] = xrefEntry{off: a, stream: true, idx: int(b)}
			}
		}
	}
	return s.Dict, nil
}

var objHeader = regexp.MustCompile(`(\d+)[ \t\r\n\f\x00]+(\d+)[ \t\r\n\f\x00]+obj\b`)

// rebuildXref finds every "n g obj" in the file; the last definition of a
// number wins, as with incremental updates. The trailer is the last
// trailer dictionary, or the catalog found among the objects.
func (f *File) rebuildXref() error {
	for _, m := range objHeader.FindAllSubmatchIndex(f.data, -1) {
		if m[0] > 0 && !isSpace(f.data[m[0]-1]) && !isDelim(f.data[m[0]-1]) {
			continue
		}
		num, _ := strconv.Atoi(string(f.data[m[2]:m[3]]))
		f.xref[num] = xrefEntry{off: int64(m[0])}
	}
	if len(f.xref) == 0 {
		return errors.New("pdf: no objects")
	}
	for p := len(f.data); ; {
		i := bytes.LastIndex(f.data[:p], []byte("trailer"))
		if i < 0 {
			break
		}
		l := &lexer{b: f.data, pos: i + len("trailer")}
		if d, ok := l.object(0).(Dict); ok && d["Root"] != nil {
			f.trailer = d
			break
		}
		p = i
	}
	// Objects kept in object streams, and xref streams' trailers.
	for num := range f.xref {
		obj := f.Resolve(Ref{num, 0})
		s, ok := obj.(*Stream)
		if !ok {
			continue
		}
		switch s.Dict["Type"] {
		case Name("XRef"):
			if f.trailer == nil && s.Dict["Root"] != nil {
				f.trailer = s.Dict
			}
		case Name("ObjStm"):
			for i, n := range f.objStmNumbers(s) {
				if _, known := f.xref[n]; !known {
					f.xref[n] = xrefEntry{off: int64(num), stream: true, idx: i}
				}
			}
		}
	}
	if f.trailer == nil {
		for num := range f.xref {
			if d := f.dict(Ref{num, 0}); d != nil && d["Type"] == Name("Catalog") {
				f.trailer = Dict{"Root": Ref{num, 0}}
				break
			}
		}
	}
	if f.trailer == nil {
		return errors.New("pdf: no trailer and no catalog")
	}
	return nil
}

// objectAt reads "n g obj ... endobj" at off.
func (f *File) objectAt(off int, decrypt bool) (Ref, any, error) {
	l := &lexer{b: f.data, pos: off}
	num, ok1 := l.token().(int64)
	gen, ok2 := l.token().(int64)
	k, ok3 := l.token().(keyword)
	if !ok1 || !ok2 || !ok3 || k != "obj" {
		return Ref{}, nil, fmt.Errorf("pdf: no object at offset %d", off)
	}
	ref := Ref{int(num), int(gen)}
	obj := l.object(0)
	save := l.pos
	if k, ok := l.token().(keyword); ok && k == "stream" {
		d, _ := obj.(Dict)
		if d == nil {
			d = Dict{}
		}
		if l.pos < len(f.data) && f.data[l.pos] == '\r' {
			l.pos++
		}
		if l.pos < len(f.data) && f.data[l.pos] == '\n' {
			l.pos++
		}
		start := l.pos
		end := -1
		if n, ok := f.Resolve(d["Length"]).(int64); ok && n >= 0 && start+int(n) <= len(f.data) {
			after := &lexer{b: f.data, pos: start + int(n)}
			if k, ok := after.token().(keyword); ok && k == "endstream" {
				end = start + int(n)
			}
		}
		if end < 0 {
			// A wrong /Length: the data ends at "endstream".
			i := bytes.Index(f.data[start:], []byte("endstream"))
			if i < 0 {
				end = len(f.data)
			} else {
				end = start + i
				for end > start && (f.data[end-1] == '\n' || f.data[end-1] == '\r') {
					end--
				}
			}
		}
		s := &Stream{Dict: d, raw: f.data[start:end], ref: ref}
		if decrypt && f.crypt != nil {
			f.crypt.decryptStrings(s.Dict, ref)
		}
		return ref, s, nil
	}
	l.pos = save
	if decrypt && f.crypt != nil {
		obj = f.crypt.decryptObject(obj, ref)
	}
	return ref, obj, nil
}

func (f *File) objStmNumbers(s *Stream) []int {
	data, err := f.decode(s)
	if err != nil {
		return nil
	}
	n, _ := s.Dict["N"].(int64)
	l := &lexer{b: data}
	var nums []int
	for range n {
		num, ok := l.token().(int64)
		if !ok {
			break
		}
		l.token()
		nums = append(nums, int(num))
	}
	return nums
}

// Resolve follows references: it returns the object o refers to, or o.
func (f *File) Resolve(o any) any {
	for range 32 {
		r, ok := o.(Ref)
		if !ok {
			return o
		}
		o = f.load(r.Num)
	}
	return nil
}

func (f *File) load(num int) any {
	if o, ok := f.cache[num]; ok {
		return o
	}
	if f.loading[num] {
		return nil
	}
	f.loading[num] = true
	defer delete(f.loading, num)
	e, ok := f.xref[num]
	var obj any
	switch {
	case !ok || e.off < 0:
	case e.stream:
		obj = f.fromObjStm(int(e.off), e.idx, num)
	default:
		if ref, o, err := f.objectAt(int(e.off), true); err == nil && ref.Num == num {
			obj = o
		}
	}
	f.cache[num] = obj
	return obj
}

func (f *File) fromObjStm(stm, idx, num int) any {
	s, ok := f.Resolve(Ref{stm, 0}).(*Stream)
	if !ok {
		return nil
	}
	data, err := f.decode(s)
	if err != nil {
		return nil
	}
	n, _ := s.Dict["N"].(int64)
	first, _ := s.Dict["First"].(int64)
	l := &lexer{b: data}
	off := -1
	for i := range int(n) {
		o, ok1 := l.token().(int64)
		p, ok2 := l.token().(int64)
		if !ok1 || !ok2 {
			break
		}
		if int(o) == num || i == idx && off < 0 {
			off = int(p)
			if int(o) == num {
				break
			}
		}
	}
	if off < 0 || int(first)+off >= len(data) {
		return nil
	}
	l = &lexer{b: data, pos: int(first) + off}
	return l.object(0)
}

func (f *File) dict(o any) Dict {
	switch v := f.Resolve(o).(type) {
	case Dict:
		return v
	case *Stream:
		return v.Dict
	}
	return nil
}

func (f *File) array(o any) Array {
	a, _ := f.Resolve(o).(Array)
	return a
}

func (f *File) num(o any) (float64, bool) {
	switch v := f.Resolve(o).(type) {
	case int64:
		return float64(v), true
	case float64:
		return v, true
	}
	return 0, false
}

func (f *File) name(o any) Name {
	n, _ := f.Resolve(o).(Name)
	return n
}

// decode returns the data of a stream with its filters undone.
func (f *File) decode(s *Stream) ([]byte, error) {
	data := s.raw
	if f.crypt != nil && s.ref.Num != 0 && s.Dict["Type"] != Name("XRef") {
		data = f.crypt.decryptStream(s, data)
	}
	filters := f.Resolve(s.Dict["Filter"])
	params := f.Resolve(s.Dict["DecodeParms"])
	if params == nil {
		params = f.Resolve(s.Dict["DP"])
	}
	var fs Array
	var ps Array
	switch v := filters.(type) {
	case Name:
		fs = Array{v}
		ps = Array{params}
	case Array:
		fs = v
		ps, _ = params.(Array)
	}
	for i, fl := range fs {
		var p Dict
		if i < len(ps) {
			p = f.dict(ps[i])
		}
		var err error
		data, err = f.filter(f.name(fl), p, data)
		if err != nil {
			return data, err
		}
	}
	return data, nil
}
