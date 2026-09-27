// Package safetensors lit et écrit le format safetensors : un en-tête JSON
// décrivant chaque tenseur, suivi des données brutes en little-endian.
//
// Les tenseurs sont toujours rendus en float32 : c'est la précision de
// calcul d'indecis. BF16 et F16 sont convertis à la lecture.
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

// Tensor est un tenseur décodé.
type Tensor struct {
	Shape []int
	Data  []float32
	// DType est le format d'écriture : "F32" (défaut) ou "BF16", arrondi au
	// plus proche. À la lecture, il vaut toujours "".
	DType string
}

// Len retourne le nombre d'éléments attendu d'après la forme.
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

// maxHeader borne l'en-tête : un fichier corrompu ne doit pas provoquer une
// allocation de plusieurs gigaoctets.
const maxHeader = 100 << 20

// ReadFile lit tous les tenseurs d'un fichier.
func ReadFile(path string) (map[string]Tensor, map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return Parse(b)
}

// Parse décode un contenu safetensors complet.
func Parse(b []byte) (map[string]Tensor, map[string]string, error) {
	f, err := parseHeader(b)
	if err != nil {
		return nil, nil, err
	}
	out := make(map[string]Tensor, len(f.entries))
	for name, e := range f.entries {
		t, err := decode(e, f.data)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: %s : %w", name, err)
		}
		out[name] = t
	}
	return out, f.meta, nil
}

// File est un fichier safetensors dont les tenseurs sont décodés à la
// demande. Ouvert par Open, son contenu est projeté en mémoire quand le
// système le permet : les tenseurs laissés bruts (Raw) ne coûtent alors que
// les pages effectivement lues, partagées entre processus.
//
// Le fichier ne doit pas être modifié en place tant que File est utilisé :
// le remplacer par un renommage est sans danger.
type File struct {
	entries map[string]entry
	meta    map[string]string
	data    []byte
	mapped  bool // data est une projection du fichier
}

// Open ouvre un fichier safetensors sans décoder ses tenseurs.
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

// Evict libère les pages d'un tenseur déjà décodé (Tensor en a fait une
// copie) : sans effet sur le contenu, qui sera relu dans le fichier si
// besoin. N'agit que sur un fichier projeté en mémoire.
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
		return nil, fmt.Errorf("safetensors: fichier tronqué")
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if n > maxHeader || 8+n > uint64(len(b)) {
		return nil, fmt.Errorf("safetensors: en-tête de %d octets invalide", n)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &raw); err != nil {
		return nil, fmt.Errorf("safetensors: en-tête : %w", err)
	}
	f := &File{entries: make(map[string]entry, len(raw)), data: b[8+n:]}
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &f.meta); err != nil {
				return nil, fmt.Errorf("safetensors: métadonnées : %w", err)
			}
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("safetensors: %s : %w", name, err)
		}
		if _, _, err := e.bytes(f.data); err != nil {
			return nil, fmt.Errorf("safetensors: %s : %w", name, err)
		}
		f.entries[name] = e
	}
	return f, nil
}

// Names liste les tenseurs du fichier, triés.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.entries))
	for name := range f.entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Meta retourne les métadonnées du fichier.
func (f *File) Meta() map[string]string { return f.meta }

// Tensor décode un tenseur en float32.
func (f *File) Tensor(name string) (Tensor, bool, error) {
	e, ok := f.entries[name]
	if !ok {
		return Tensor{}, false, nil
	}
	t, err := decode(e, f.data)
	if err != nil {
		return t, true, fmt.Errorf("safetensors: %s : %w", name, err)
	}
	return t, true, nil
}

// Raw retourne un tenseur sans le décoder : son dtype, sa forme et ses
// octets en little-endian, à ne pas modifier.
func (f *File) Raw(name string) (dtype string, shape []int, data []byte, ok bool) {
	e, ok := f.entries[name]
	if !ok {
		return "", nil, nil, false
	}
	_, raw, _ := e.bytes(f.data)
	return e.DType, e.Shape, raw, true
}

// bytes retourne la taille d'un élément et les octets du tenseur, après
// avoir vérifié qu'ils correspondent à la forme.
func (e entry) bytes(data []byte) (int, []byte, error) {
	n := 1
	for _, d := range e.Shape {
		if d < 0 {
			return 0, nil, fmt.Errorf("forme %v invalide", e.Shape)
		}
		n *= d
	}
	start, end := e.Offsets[0], e.Offsets[1]
	if start < 0 || end < start || end > len(data) {
		return 0, nil, fmt.Errorf("offsets [%d, %d) hors des données (%d octets)", start, end, len(data))
	}
	var size int
	switch e.DType {
	case "F32":
		size = 4
	case "BF16", "F16":
		size = 2
	default:
		return 0, nil, fmt.Errorf("dtype %s non pris en charge", e.DType)
	}
	raw := data[start:end]
	if len(raw) != n*size {
		return 0, nil, fmt.Errorf("%d octets pour %d éléments %s", len(raw), n, e.DType)
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
	case exp == 0: // sous-normal
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

// Write écrit des tenseurs float32. Les noms sont triés : même contenu, même
// fichier.
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
			return fmt.Errorf("safetensors: %s : %d éléments pour la forme %v", name, len(t.Data), t.Shape)
		}
		size, dtype := 4, "F32"
		switch t.DType {
		case "", "F32":
		case "BF16":
			size, dtype = 2, "BF16"
		default:
			return fmt.Errorf("safetensors: %s : dtype d'écriture %q non pris en charge", name, t.DType)
		}
		header[name] = entry{DType: dtype, Shape: t.Shape, Offsets: [2]int{offset, offset + size*len(t.Data)}}
		offset += size * len(t.Data)
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// L'en-tête est complété par des espaces pour aligner les données sur 8.
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
		bf16 := tensors[name].DType == "BF16"
		for _, v := range tensors[name].Data {
			if bf16 {
				buf = binary.LittleEndian.AppendUint16(buf, ToBF16(v))
			} else {
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

// ToBF16 arrondit un float32 au bfloat16 le plus proche (égalité : pair).
func ToBF16(v float32) uint16 {
	b := math.Float32bits(v)
	if v != v { // NaN reste NaN
		return uint16(b>>16) | 0x40
	}
	b += 0x7fff + ((b >> 16) & 1)
	return uint16(b >> 16)
}

// FromBF16 est l'inverse exact de ToBF16 sur les valeurs représentables.
func FromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
