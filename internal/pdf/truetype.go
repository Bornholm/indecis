package pdf

import "encoding/binary"

// trueTypeText maps the glyph indices of a TrueType font program to the
// text they draw, from its Unicode cmap subtables, then from the glyph
// names of its post table. nil when the font says nothing.
func trueTypeText(data []byte) map[int]string {
	tables := map[string][]byte{}
	if len(data) < 12 {
		return nil
	}
	n := int(binary.BigEndian.Uint16(data[4:]))
	for i := range n {
		p := 12 + 16*i
		if p+16 > len(data) {
			break
		}
		off := int(binary.BigEndian.Uint32(data[p+8:]))
		size := int(binary.BigEndian.Uint32(data[p+12:]))
		if off < 0 || size < 0 || off+size > len(data) || off+size < off {
			continue
		}
		tables[string(data[p:p+4])] = data[off : off+size]
	}
	out := map[int]string{}
	if c := tables["cmap"]; len(c) >= 4 {
		n := int(binary.BigEndian.Uint16(c[2:]))
		for i := range n {
			p := 4 + 8*i
			if p+8 > len(c) {
				break
			}
			platform := binary.BigEndian.Uint16(c[p:])
			enc := binary.BigEndian.Uint16(c[p+2:])
			off := int(binary.BigEndian.Uint32(c[p+4:]))
			if !(platform == 0 || platform == 3 && (enc == 1 || enc == 10)) || off >= len(c) {
				continue
			}
			cmapSubtable(c[off:], out)
		}
	}
	if p := tables["post"]; len(p) >= 34 && binary.BigEndian.Uint32(p) == 0x00020000 {
		postNames(p, out)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func cmapSubtable(t []byte, out map[int]string) {
	if len(t) < 4 {
		return
	}
	u16 := func(p int) int {
		if p+2 > len(t) {
			return 0
		}
		return int(binary.BigEndian.Uint16(t[p:]))
	}
	u32 := func(p int) int {
		if p+4 > len(t) {
			return 0
		}
		return int(binary.BigEndian.Uint32(t[p:]))
	}
	set := func(gid int, r rune) {
		if gid == 0 || r == 0 || r == 0xFFFF {
			return
		}
		// A glyph several characters map to keeps the first.
		if _, ok := out[gid]; !ok {
			out[gid] = string(r)
		}
	}
	switch u16(0) {
	case 4:
		segs := u16(6) / 2
		ends, starts, deltas, offs := 14, 16+2*segs, 16+4*segs, 16+6*segs
		for s := range segs {
			end, start := u16(ends+2*s), u16(starts+2*s)
			delta, ro := u16(deltas+2*s), u16(offs+2*s)
			if start > end || end-start > 0x10000 {
				continue
			}
			for c := start; c <= end && c != 0xFFFF; c++ {
				var gid int
				if ro == 0 {
					gid = (c + delta) & 0xFFFF
				} else {
					p := offs + 2*s + ro + 2*(c-start)
					gid = u16(p)
					if gid != 0 {
						gid = (gid + delta) & 0xFFFF
					}
				}
				set(gid, rune(c))
			}
		}
	case 12:
		n := u32(12)
		for g := range n {
			p := 16 + 12*g
			if p+12 > len(t) {
				break
			}
			start, end, gid := u32(p), u32(p+4), u32(p+8)
			for c := start; c <= end && c-start < 0x10000 && c <= 0x10FFFF; c++ {
				set(gid+c-start, rune(c))
			}
		}
	case 6:
		first, n := u16(6), u16(8)
		for i := range n {
			set(u16(10+2*i), rune(first+i))
		}
	}
}

func postNames(p []byte, out map[int]string) {
	n := int(binary.BigEndian.Uint16(p[32:]))
	idx := 34
	if idx+2*n > len(p) {
		return
	}
	var custom []string
	for q := idx + 2*n; q < len(p); {
		l := int(p[q])
		if q+1+l > len(p) {
			break
		}
		custom = append(custom, string(p[q+1:q+1+l]))
		q += 1 + l
	}
	for gid := range n {
		k := int(binary.BigEndian.Uint16(p[idx+2*gid:]))
		if _, ok := out[gid]; ok || k < 258 {
			// Standard Macintosh names: the glyph list knows them, but fonts
			// that use them also have a cmap.
			continue
		}
		if k-258 < len(custom) {
			if t := glyphText(custom[k-258]); t != "" {
				out[gid] = t
			}
		}
	}
}
