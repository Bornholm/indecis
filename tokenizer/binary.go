package tokenizer

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/bornholm/indecis/internal/mapfile"
)

// A tokenizer.bin holds the tables of a tokenizer as they sit in memory,
// next to the tokenizer.json they come from. Load maps it instead of
// parsing the JSON: loading takes a few milliseconds instead of half a
// second for 256,000 tokens, allocates nothing for the tables (13 MB of
// heap otherwise), and the pages read are those of the file, shared
// between the processes that serve the same model.
//
// Layout, in the byte order of the machine that wrote it:
//
//	magic    8 bytes, binMagic
//	order    uint32, 0x01020304 as written
//	reserved uint32
//	source   32 bytes, SHA-256 of the tokenizer.json
//	sections 6 × (offset, length) uint64: meta (JSON), blob, off, slots,
//	         keys, vals, each starting on a multiple of 8
//
// The file is used only if its source digest is that of the
// tokenizer.json beside it: a JSON modified afterwards is read again. The
// digest does not cover the code that wrote it: a change of layout, of
// strHash or of merge must change binMagic, so that older files are
// rejected. The content of the sections is trusted: a damaged file can make
// Encode panic, not read out of bounds.

const (
	binMagic    = "INDTOK01"
	binOrder    = 0x01020304
	binSections = 6
	binHeader   = 8 + 4 + 4 + sha256.Size + binSections*16
)

// BinaryPath is the path of the tokenizer.bin that goes with a
// tokenizer.json.
func BinaryPath(jsonPath string) string {
	return strings.TrimSuffix(jsonPath, ".json") + ".bin"
}

// binMeta holds the small part of a Tokenizer.
type binMeta struct {
	Size  int              `json:"size"`
	Raw   bool             `json:"raw,omitempty"`
	Unk   int32            `json:"unk"`
	Bos   int32            `json:"bos"`
	Eos   int32            `json:"eos"`
	Pad   int32            `json:"pad"`
	Bytes [256]int32       `json:"bytes"`
	Names map[int32]string `json:"names,omitempty"`
	Added []binAdded       `json:"added,omitempty"`
}

type binAdded struct {
	Content string `json:"content"`
	ID      int32  `json:"id"`
	LStrip  bool   `json:"lstrip,omitempty"`
	RStrip  bool   `json:"rstrip,omitempty"`
}

// WriteBinary writes the tokenizer.bin of a tokenizer.json (BinaryPath),
// which Load and LoadShared then map instead of parsing the JSON.
func WriteBinary(jsonPath string) error {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return err
	}
	t, err := Parse(data)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := t.writeBinary(&buf, sha256.Sum256(data)); err != nil {
		return err
	}
	out := BinaryPath(jsonPath)
	tmp, err := os.CreateTemp(filepath.Dir(out), ".tokenizer-*.bin")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(buf.Bytes()); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), out)
}

func (t *Tokenizer) writeBinary(w io.Writer, source [sha256.Size]byte) error {
	meta := binMeta{Size: t.size, Raw: t.raw, Unk: t.unk, Bos: t.bos, Eos: t.eos, Pad: t.pad, Bytes: t.bytes}
	if len(t.names) > 0 {
		meta.Names = t.names
	}
	for _, list := range t.added {
		for _, a := range list {
			meta.Added = append(meta.Added, binAdded{Content: a.content, ID: a.id, LStrip: a.lstrip, RStrip: a.rstrip})
		}
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	sections := [binSections][]byte{
		metaJSON,
		unsafe.Slice(unsafe.StringData(t.vocab.blob), len(t.vocab.blob)),
		asBytes(t.vocab.off),
		asBytes(t.vocab.slots),
		asBytes(t.merges.keys),
		asBytes(t.merges.vals),
	}
	head := make([]byte, binHeader)
	copy(head, binMagic)
	binary.NativeEndian.PutUint32(head[8:], binOrder)
	copy(head[16:], source[:])
	pos := uint64(binHeader)
	for i, s := range sections {
		pos = align8(pos)
		binary.NativeEndian.PutUint64(head[16+sha256.Size+i*16:], pos)
		binary.NativeEndian.PutUint64(head[16+sha256.Size+i*16+8:], uint64(len(s)))
		pos += uint64(len(s))
	}
	if _, err := w.Write(head); err != nil {
		return err
	}
	pos = binHeader
	var pad [8]byte
	for _, s := range sections {
		if n := align8(pos) - pos; n > 0 {
			if _, err := w.Write(pad[:n]); err != nil {
				return err
			}
			pos += n
		}
		if _, err := w.Write(s); err != nil {
			return err
		}
		pos += uint64(len(s))
	}
	return nil
}

func align8(n uint64) uint64 { return (n + 7) &^ 7 }

func asBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var z T
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*int(unsafe.Sizeof(z)))
}

func fromBytes[T any](b []byte) []T {
	var z T
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/int(unsafe.Sizeof(z)))
}

var errStale = errors.New("tokenizer: tokenizer.bin from another tokenizer.json")

// loadBinary maps the tokenizer.bin at path if its source digest is
// source; errStale if it comes from another JSON.
func loadBinary(path string, source [sha256.Size]byte) (t *Tokenizer, err error) {
	b, mapped, err := mapfile.Map(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			mapfile.Unmap(b, mapped)
		}
	}()
	if len(b) < binHeader || string(b[:8]) != binMagic {
		return nil, fmt.Errorf("tokenizer: %s: not a tokenizer.bin", path)
	}
	if binary.NativeEndian.Uint32(b[8:]) != binOrder {
		return nil, fmt.Errorf("tokenizer: %s: written with another byte order", path)
	}
	if !bytes.Equal(b[16:16+sha256.Size], source[:]) {
		return nil, errStale
	}
	var sec [binSections][]byte
	for i := range sec {
		off := binary.NativeEndian.Uint64(b[16+sha256.Size+i*16:])
		n := binary.NativeEndian.Uint64(b[16+sha256.Size+i*16+8:])
		if off%8 != 0 || off > uint64(len(b)) || n > uint64(len(b))-off {
			return nil, fmt.Errorf("tokenizer: %s: section %d out of the file", path, i)
		}
		sec[i] = b[off : off+n]
	}
	var meta binMeta
	if err := json.Unmarshal(sec[0], &meta); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	t = &Tokenizer{
		vocab: &strTable{
			blob:  unsafe.String(unsafe.SliceData(sec[1]), len(sec[1])),
			off:   fromBytes[uint32](sec[2]),
			slots: fromBytes[int32](sec[3]),
		},
		merges: mergeTable{keys: fromBytes[uint64](sec[4]), vals: fromBytes[merge](sec[5])},
		size:   meta.Size,
		raw:    meta.Raw,
		unk:    meta.Unk,
		bos:    meta.Bos,
		eos:    meta.Eos,
		pad:    meta.Pad,
		bytes:  meta.Bytes,
		names:  meta.Names,
		added:  make(map[byte][]addedToken),
		cache:  make(map[string][]int32),
	}
	if mapped {
		t.mapping = b
	}
	slots := len(t.vocab.slots)
	switch {
	case len(t.vocab.off) != meta.Size+1 || int(t.vocab.off[meta.Size]) != len(t.vocab.blob):
		return nil, fmt.Errorf("tokenizer: %s: inconsistent vocabulary", path)
	case slots == 0 || slots&(slots-1) != 0 || slots < 2*meta.Size:
		return nil, fmt.Errorf("tokenizer: %s: inconsistent string table", path)
	case len(t.merges.keys) != len(t.merges.vals):
		return nil, fmt.Errorf("tokenizer: %s: inconsistent merges", path)
	}
	t.vocab.mask = uint64(slots - 1)
	if t.names == nil {
		t.names = make(map[int32]string)
	}
	// The lists of added tokens were written one after the other, each in
	// its order (longest first, see read): appending rebuilds them as is.
	for _, a := range meta.Added {
		if a.Content == "" {
			return nil, fmt.Errorf("tokenizer: %s: empty added token", path)
		}
		first := a.Content[0]
		t.added[first] = append(t.added[first], addedToken{content: a.Content, id: a.ID, lstrip: a.LStrip, rstrip: a.RStrip})
	}
	return t, nil
}
