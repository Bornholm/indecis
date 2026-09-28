// Package safetensors reads and writes the safetensors format: a JSON
// header describing each tensor, followed by raw little-endian data.
//
// Tensors are always returned as float32: that is indecis's compute
// precision. BF16 and F16 are converted on read.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// Tensor is a decoded tensor.
type Tensor struct {
	Shape []int
	Data  []float32
	// DType is the write format: "F32" (default), "BF16" (rounded to
	// nearest), or "I8" (Data holds already-quantized integers). On read,
	// it is always "".
	DType string
}

// Len returns the number of elements expected from the shape.
func (t Tensor) Len() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

type entry struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// maxHeader bounds the header: a corrupted file must not trigger a
// multi-gigabyte allocation.
const maxHeader = 100 << 20

// ReadFile reads all tensors of a file.
func ReadFile(path string) (map[string]Tensor, map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return Parse(b)
}

// Parse decodes complete safetensors content.
func Parse(b []byte) (map[string]Tensor, map[string]string, error) {
	f, err := parseHeader(b)
	if err != nil {
		return nil, nil, err
	}
	out := make(map[string]Tensor, len(f.entries))
	for name, e := range f.entries {
		t, err := decode(e, f.data)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: %s: %w", name, err)
		}
		out[name] = t
	}
	return out, f.meta, nil
}

// File is a safetensors file whose tensors are decoded on demand. Opened by
// Open, its content is mapped into memory when the system allows it: the
// tensors left raw (Raw) then cost only the pages actually read, shared
// between processes.
//
// The file must not be modified in place while File is in use: replacing it
// via a rename is safe.
type File struct {
	entries map[string]entry
	meta    map[string]string
	data    []byte
	mapped  bool // data is a mapping of the file
}

// Open opens a safetensors file without decoding its tensors.
func Open(path string) (*File, error) {
	b, mapped, err := mapFile(path)
	if err != nil {
		return nil, err
	}
	f, err := parseHeader(b)
	if err != nil {
		return nil, err
	}
	f.mapped = mapped
	return f, nil
}

// Evict releases the pages of a tensor already decoded (Tensor made a copy
// of it): no effect on the content, which will be reread from the file if
// needed. Only affects a file mapped into memory.
func (f *File) Evict(name string) {
	if !f.mapped {
		return
	}
	if e, ok := f.entries[name]; ok {
		if _, raw, err := e.bytes(f.data); err == nil {
			evict(raw)
		}
	}
}

func parseHeader(b []byte) (*File, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("safetensors: truncated file")
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if n > maxHeader || 8+n > uint64(len(b)) {
		return nil, fmt.Errorf("safetensors: invalid %d-byte header", n)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &raw); err != nil {
		return nil, fmt.Errorf("safetensors: header: %w", err)
	}
	f := &File{entries: make(map[string]entry, len(raw)), data: b[8+n:]}
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &f.meta); err != nil {
				return nil, fmt.Errorf("safetensors: metadata: %w", err)
			}
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("safetensors: %s: %w", name, err)
		}
		if _, _, err := e.bytes(f.data); err != nil {
			return nil, fmt.Errorf("safetensors: %s: %w", name, err)
		}
		f.entries[name] = e
	}
	return f, nil
}

// Names lists the tensors of the file, sorted.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.entries))
	for name := range f.entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Meta returns the metadata of the file.
func (f *File) Meta() map[string]string { return f.meta }

// Tensor decodes a tensor as float32.
func (f *File) Tensor(name string) (Tensor, bool, error) {
	e, ok := f.entries[name]
	if !ok {
		return Tensor{}, false, nil
	}
	t, err := decode(e, f.data)
	if err != nil {
		return t, true, fmt.Errorf("safetensors: %s: %w", name, err)
	}
	return t, true, nil
}

// Raw returns a tensor without decoding it: its dtype, its shape, and its
// little-endian bytes, not to be modified.
func (f *File) Raw(name string) (dtype string, shape []int, data []byte, ok bool) {
	e, ok := f.entries[name]
	if !ok {
		return "", nil, nil, false
	}
	_, raw, _ := e.bytes(f.data)
	return e.DType, e.Shape, raw, true
}

// bytes returns the size of an element and the bytes of the tensor, after
// checking that they match the shape.
func (e entry) bytes(data []byte) (int, []byte, error) {
	n := 1
	for _, d := range e.Shape {
		if d < 0 {
			return 0, nil, fmt.Errorf("invalid shape %v", e.Shape)
		}
		n *= d
	}
	start, end := e.Offsets[0], e.Offsets[1]
	if start < 0 || end < start || end > len(data) {
		return 0, nil, fmt.Errorf("offsets [%d, %d) out of data bounds (%d bytes)", start, end, len(data))
	}
	var size int
	switch e.DType {
	case "F32":
		size = 4
	case "BF16", "F16":
		size = 2
	case "I8":
		size = 1
	default:
		return 0, nil, fmt.Errorf("unsupported dtype %s", e.DType)
	}
	raw := data[start:end]
	if len(raw) != n*size {
		return 0, nil, fmt.Errorf("%d bytes for %d %s elements", len(raw), n, e.DType)
	}
	return size, raw, nil
}

func decode(e entry, data []byte) (Tensor, error) {
	t := Tensor{Shape: e.Shape}
	_, raw, err := e.bytes(data)
	if err != nil {
		return t, err
	}
	t.Data = make([]float32, t.Len())
	switch e.DType {
	case "F32":
		for i := range t.Data {
			t.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case "BF16":
		for i := range t.Data {
			t.Data[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16)
		}
	case "F16":
		for i := range t.Data {
			t.Data[i] = halfToFloat(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case "I8":
		for i := range t.Data {
			t.Data[i] = float32(int8(raw[i]))
		}
	}
	return t, nil
}

func halfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal
		f := float32(mant) / 1024 / 16384
		if sign != 0 {
			return -f
		}
		return f
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

// Write writes float32 tensors. Names are sorted: same content, same
// file.
func Write(w io.Writer, tensors map[string]Tensor, meta map[string]string) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	header := make(map[string]any, len(tensors)+1)
	if len(meta) > 0 {
		header["__metadata__"] = meta
	}
	offset := 0
	for _, name := range names {
		t := tensors[name]
		if len(t.Data) != t.Len() {
			return fmt.Errorf("safetensors: %s: %d elements for shape %v", name, len(t.Data), t.Shape)
		}
		size, dtype := 4, "F32"
		switch t.DType {
		case "", "F32":
		case "BF16":
			size, dtype = 2, "BF16"
		case "I8":
			size, dtype = 1, "I8"
		default:
			return fmt.Errorf("safetensors: %s: unsupported write dtype %q", name, t.DType)
		}
		header[name] = entry{DType: dtype, Shape: t.Shape, Offsets: [2]int{offset, offset + size*len(t.Data)}}
		offset += size * len(t.Data)
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// The header is padded with spaces to align the data on 8 bytes.
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hb)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := w.Write(hb); err != nil {
		return err
	}
	buf := make([]byte, 0, 1<<16)
	for _, name := range names {
		dt := tensors[name].DType
		for _, v := range tensors[name].Data {
			switch dt {
			case "BF16":
				buf = binary.LittleEndian.AppendUint16(buf, ToBF16(v))
			case "I8":
				buf = append(buf, byte(int8(max(-128, min(127, v)))))
			default:
				buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
			}
			if len(buf) >= cap(buf)-4 {
				if _, err := w.Write(buf); err != nil {
					return err
				}
				buf = buf[:0]
			}
		}
	}
	_, err = w.Write(buf)
	return err
}

// ToBF16 rounds a float32 to the nearest bfloat16 (ties to even).
func ToBF16(v float32) uint16 {
	b := math.Float32bits(v)
	if v != v { // NaN stays NaN
		return uint16(b>>16) | 0x40
	}
	b += 0x7fff + ((b >> 16) & 1)
	return uint16(b >> 16)
}

// FromBF16 is the exact inverse of ToBF16 on representable values.
func FromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
