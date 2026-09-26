package indecis

import (
	"encoding/binary"
	"fmt"
	"math"
)

// mappedEmbeddings lit la table d'embeddings d'un modèle sauvegardé
// directement dans le fichier projeté en mémoire : bf16 pour les lignes
// d'origine, float32 pour celles que l'entraînement a modifiées. La table
// occuperait 393 Mo en float32 ; seules les pages des tokens rencontrés
// sont chargées.
type mappedEmbeddings struct {
	h     int
	bf16  []byte        // [V, H] en bf16 little-endian
	exact map[int32]int // id → rang dans values
	f32   []byte        // [len(exact), H] en float32 little-endian
}

func newMappedEmbeddings(h int, bf16 []byte, rows []float32, values []byte) (*mappedEmbeddings, error) {
	e := &mappedEmbeddings{h: h, bf16: bf16, exact: make(map[int32]int, len(rows)), f32: values}
	vocab := len(bf16) / (2 * h)
	if len(values) != len(rows)*h*4 {
		return nil, fmt.Errorf("indecis: lignes exactes mal formées")
	}
	for i, r := range rows {
		id := int(r)
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("indecis: ligne exacte %d hors vocabulaire", id)
		}
		e.exact[int32(id)] = i
	}
	return e, nil
}

func (e *mappedEmbeddings) Row(id int32, dst []float32) {
	h := e.h
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
