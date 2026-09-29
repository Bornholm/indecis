package siglip

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// Text is the text tower: token and position embeddings, transformer
// layers, a final LayerNorm, and a projection of the last position.
type Text struct {
	cfg   Config
	work  *workPool
	table []byte // token embeddings [Vocab, H], F32, read in place
	pos   []float32
	enc   *encoder
	final layerNorm
	head  *linear
}

func loadText(cfg Config, w weights, int8 bool) (*Text, error) {
	H := cfg.Hidden
	t := &Text{cfg: cfg, work: &workPool{cfg: cfg, T: cfg.TextLen}}
	dtype, shape, raw, ok := w.f.Raw("text_model.embeddings.token_embedding.weight")
	if !ok {
		return nil, fmt.Errorf("siglip: token embeddings missing")
	}
	if dtype != "F32" || len(shape) != 2 || shape[0] != cfg.Vocab || shape[1] != H {
		return nil, fmt.Errorf("siglip: token embeddings %s %v, expected F32 [%d %d]", dtype, shape, cfg.Vocab, H)
	}
	// 256,000 × 768 floats: memory-mapped, only the rows of the tokens met
	// are read.
	t.table = raw
	var err error
	if t.pos, err = w.get("text_model.embeddings.position_embedding.weight", cfg.TextLen, H); err != nil {
		return nil, err
	}
	if t.enc, err = loadEncoder(cfg, w, "text_model.encoder", int8); err != nil {
		return nil, err
	}
	if t.final, err = loadNorm(w, "text_model.final_layer_norm", H); err != nil {
		return nil, err
	}
	if t.head, err = loadLinear(w, "text_model.head", H, H, false); err != nil {
		return nil, err
	}
	return t, nil
}

// Embed computes the embedding of a tokenized text, exactly TextLen ids
// (padded, see Pad). The result, [H], is not normalized.
func (t *Text) Embed(ids []int32) ([]float32, error) {
	cfg := t.cfg
	T, H := cfg.TextLen, cfg.Hidden
	if len(ids) != T {
		return nil, fmt.Errorf("siglip: %d ids, expected %d", len(ids), T)
	}
	w := t.work.get()
	defer t.work.put(w)
	x := w.x
	for i, id := range ids {
		if id < 0 || int(id) >= cfg.Vocab {
			return nil, fmt.Errorf("siglip: id %d out of vocabulary", id)
		}
		row := t.table[int(id)*H*4:]
		for j := 0; j < H; j++ {
			x[i*H+j] = math.Float32frombits(binary.LittleEndian.Uint32(row[j*4:]))
		}
	}
	linalg.AddTo(x, t.pos)
	t.enc.forward(x, T, w.b)
	// Only the last position is used: normalize it alone.
	last := x[(T-1)*H:]
	n := make([]float32, H)
	t.final.apply(n, last, 1, H, cfg.Eps)
	out := make([]float32, H)
	t.head.apply(out, n, 1, 1)
	return out, nil
}

// Pad frames tokenized text to the tower's fixed length: truncated to
// TextLen ids keeping the final <eos>, then padded with pad. SigLIP reads
// the last position, so a text always takes the whole length.
func (t *Text) Pad(ids []int32, eos, pad int32) []int32 {
	T := t.cfg.TextLen
	out := make([]int32, T)
	if len(ids) > T {
		copy(out, ids[:T])
		out[T-1] = eos
		return out
	}
	copy(out, ids)
	for i := len(ids); i < T; i++ {
		out[i] = pad
	}
	return out
}
