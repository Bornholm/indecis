package indecis

import (
	"encoding/binary"
	"fmt"
	"math"
)

// mappedEmbeddings reads the embedding table of a saved model directly from
// the memory-mapped file: bf16 for the original rows, float32 for the rows
// that training modified. The table would take 393 MB in float32; only the
// pages of the tokens encountered are loaded.
type mappedEmbeddings struct {
	h     int
	bf16  []byte        // [V, H] in bf16 little-endian
	exact map[int32]int // id -> rank in values
	f32   []byte        // [len(exact), H] in float32 little-endian

	// int8 table (WithInt8Embeddings): i8 [V, H], scale [V].
	i8    []byte
	scale []float32
}

// newInt8Embeddings reads an int8 table: each row is worth i8 * scale.
func newInt8Embeddings(h int, i8 []byte, scale []float32) (*mappedEmbeddings, error) {
	if len(i8) != len(scale)*h {
		return nil, fmt.Errorf("indecis: int8 table of %d bytes for %d rows", len(i8), len(scale))
	}
	return &mappedEmbeddings{h: h, i8: i8, scale: scale}, nil
}

func newMappedEmbeddings(h int, bf16 []byte, rows []float32, values []byte) (*mappedEmbeddings, error) {
	e := &mappedEmbeddings{h: h, bf16: bf16, exact: make(map[int32]int, len(rows)), f32: values}
	vocab := len(bf16) / (2 * h)
	if len(values) != len(rows)*h*4 {
		return nil, fmt.Errorf("indecis: malformed exact rows")
	}
	for i, r := range rows {
		id := int(r)
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("indecis: exact row %d out of vocabulary", id)
		}
		e.exact[int32(id)] = i
	}
	return e, nil
}

func (e *mappedEmbeddings) Row(id int32, dst []float32) {
	h := e.h
	if e.i8 != nil {
		raw, s := e.i8[int(id)*h:(int(id)+1)*h], e.scale[id]
		for j := range dst[:h] {
			dst[j] = float32(int8(raw[j])) * s
		}
		return
	}
	if i, ok := e.exact[id]; ok {
		raw := e.f32[i*h*4 : (i+1)*h*4]
		for j := range dst[:h] {
			dst[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
		}
		return
	}
	raw := e.bf16[int(id)*h*2 : (int(id)+1)*h*2]
	for j := range dst[:h] {
		dst[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*j:])) << 16)
	}
}
