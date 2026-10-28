package pdf

import (
	"bytes"
	"compress/flate"
	"compress/zlib"
	"errors"
	"fmt"
	"io"
)

// filter undoes one stream filter. Image filters (DCT, JPX, CCITT, JBIG2)
// leave the data as it is: images are not read.
func (f *File) filter(name Name, p Dict, data []byte) ([]byte, error) {
	switch name {
	case "FlateDecode", "Fl":
		out := inflate(data)
		return predict(f, p, out)
	case "LZWDecode", "LZW":
		early := int64(1)
		if v, ok := f.Resolve(p["EarlyChange"]).(int64); ok {
			early = v
		}
		return predict(f, p, lzw(data, early == 1))
	case "ASCIIHexDecode", "AHx":
		l := &lexer{b: append(append([]byte{'<'}, data...), '>')}
		return []byte(l.hexString()), nil
	case "ASCII85Decode", "A85":
		return ascii85(data), nil
	case "RunLengthDecode", "RL":
		return runLength(data), nil
	case "Crypt":
		return data, nil
	case "DCTDecode", "DCT", "JPXDecode", "CCITTFaxDecode", "CCF", "JBIG2Decode":
		return data, nil
	}
	return data, fmt.Errorf("pdf: unknown filter %s", name)
}

// inflate reads zlib data as far as it goes: damaged streams and streams
// with no zlib header or checksum still give what they hold.
func inflate(data []byte) []byte {
	var out bytes.Buffer
	if r, err := zlib.NewReader(bytes.NewReader(data)); err == nil {
		io.Copy(&out, r)
		if out.Len() > 0 {
			return out.Bytes()
		}
	}
	for _, skip := range []int{2, 0} {
		if len(data) < skip {
			continue
		}
		out.Reset()
		io.Copy(&out, flate.NewReader(bytes.NewReader(data[skip:])))
		if out.Len() > 0 {
			return out.Bytes()
		}
	}
	return nil
}

// predict undoes a PNG or TIFF predictor.
func predict(f *File, p Dict, data []byte) ([]byte, error) {
	if p == nil {
		return data, nil
	}
	pred, _ := f.Resolve(p["Predictor"]).(int64)
	if pred < 2 {
		return data, nil
	}
	colors, bpc, cols := int64(1), int64(8), int64(1)
	if v, ok := f.Resolve(p["Colors"]).(int64); ok && v > 0 {
		colors = v
	}
	if v, ok := f.Resolve(p["BitsPerComponent"]).(int64); ok && v > 0 {
		bpc = v
	}
	if v, ok := f.Resolve(p["Columns"]).(int64); ok && v > 0 {
		cols = v
	}
	if colors > 32 || bpc > 16 || cols > 1<<20 {
		return data, nil // not a predictor any image or xref stream uses
	}
	bpp := int(max(1, (colors*bpc+7)/8))
	row := int((colors*bpc*cols + 7) / 8)
	if pred == 2 { // TIFF, 8 bits per component only
		if bpc != 8 {
			return data, nil
		}
		for s := 0; s+row <= len(data); s += row {
			for i := bpp; i < row; i++ {
				data[s+i] += data[s+i-bpp]
			}
		}
		return data, nil
	}
	out := make([]byte, 0, len(data))
	prev := make([]byte, row)
	for s := 0; s < len(data); s += row + 1 {
		typ := data[s]
		end := min(len(data), s+1+row)
		cur := make([]byte, row)
		copy(cur, data[s+1:end])
		for i := range row {
			var a, b, c byte
			if i >= bpp {
				a = cur[i-bpp]
				c = prev[i-bpp]
			}
			b = prev[i]
			switch typ {
			case 1:
				cur[i] += a
			case 2:
				cur[i] += b
			case 3:
				cur[i] += byte((int(a) + int(b)) / 2)
			case 4:
				cur[i] += paeth(a, b, c)
			}
		}
		out = append(out, cur...)
		prev = cur
	}
	return out, nil
}

func paeth(a, b, c byte) byte {
	p := int(a) + int(b) - int(c)
	pa, pb, pc := abs(p-int(a)), abs(p-int(b)), abs(p-int(c))
	if pa <= pb && pa <= pc {
		return a
	}
	if pb <= pc {
		return b
	}
	return c
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// lzw decodes PDF's LZW, whose code width grows one code early by default
// (compress/lzw cannot do that).
func lzw(data []byte, early bool) []byte {
	var out []byte
	table := make([][]byte, 258, 4096)
	for i := range 256 {
		table[i] = []byte{byte(i)}
	}
	width := 9
	var bits uint32
	nbits := 0
	var prev []byte
	e := 0
	if early {
		e = 1
	}
	for _, c := range data {
		bits = bits<<8 | uint32(c)
		nbits += 8
		for nbits >= width {
			code := int(bits>>(nbits-width)) & (1<<width - 1)
			nbits -= width
			switch {
			case code == 256:
				table = table[:258]
				width = 9
				prev = nil
				continue
			case code == 257:
				return out
			}
			var entry []byte
			if code < len(table) {
				entry = table[code]
			} else if prev != nil {
				entry = append(append([]byte{}, prev...), prev[0])
			} else {
				return out
			}
			out = append(out, entry...)
			if prev != nil && len(table) < 4096 {
				table = append(table, append(append([]byte{}, prev...), entry[0]))
			}
			prev = entry
			if len(table)+e >= 1<<width && width < 12 {
				width++
			}
		}
	}
	return out
}

func ascii85(data []byte) []byte {
	var out []byte
	var group [5]byte
	n := 0
	for _, c := range data {
		switch {
		case c == '~':
			goto end
		case c == 'z' && n == 0:
			out = append(out, 0, 0, 0, 0)
			continue
		case c < '!' || c > 'u':
			continue
		}
		group[n] = c - '!'
		n++
		if n == 5 {
			v := uint32(0)
			for _, g := range group {
				v = v*85 + uint32(g)
			}
			out = append(out, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
			n = 0
		}
	}
end:
	if n > 1 {
		for i := n; i < 5; i++ {
			group[i] = 84
		}
		v := uint32(0)
		for _, g := range group {
			v = v*85 + uint32(g)
		}
		b := []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
		out = append(out, b[:n-1]...)
	}
	return out
}

func runLength(data []byte) []byte {
	var out []byte
	for i := 0; i < len(data); {
		n := int(data[i])
		i++
		switch {
		case n < 128:
			end := min(len(data), i+n+1)
			out = append(out, data[i:end]...)
			i = end
		case n > 128:
			if i < len(data) {
				out = append(out, bytes.Repeat(data[i:i+1], 257-n)...)
			}
			i++
		default:
			return out
		}
	}
	return out
}

var errEncrypted = errors.New("pdf: encrypted with a password")
