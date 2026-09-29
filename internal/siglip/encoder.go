package siglip

import (
	"fmt"
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// linear is y = x·Wᵀ + b, with W stored [out, in] as in PyTorch and
// packed once for the matrix products.
type linear struct {
	in, out int
	bias    []float32
	p       *linalg.PackedB
	p8      *linalg.PackedB8
}

func newLinear(w, b []float32, in, out int, int8 bool) *linear {
	l := &linear{in: in, out: out, bias: b}
	if int8 {
		l.p8 = linalg.PackB8(w, in, out, true)
	} else {
		l.p = linalg.PackB(w, in, out, true)
	}
	return l
}

// apply writes dst = x·Wᵀ + b for rows rows of x.
func (l *linear) apply(dst, x []float32, rows int) {
	if l.p8 != nil {
		linalg.MatMul8(dst[:rows*l.out], x[:rows*l.in], l.p8, rows, false)
	} else {
		linalg.MatMulPacked(dst[:rows*l.out], x[:rows*l.in], l.p, rows, false)
	}
	if l.bias != nil {
		for r := 0; r < rows; r++ {
			linalg.AddTo(dst[r*l.out:(r+1)*l.out], l.bias)
		}
	}
}

// layerNorm is a LayerNorm with scale and bias.
type layerNorm struct{ gamma, beta []float32 }

func (n layerNorm) apply(dst, x []float32, rows, width int, eps float64) {
	for r := 0; r < rows; r++ {
		o := dst[r*width : (r+1)*width]
		linalg.LayerNormRow(o, x[r*width:(r+1)*width], n.gamma, eps)
		linalg.AddTo(o, n.beta)
	}
}

func loadNorm(w weights, prefix string, width int) (layerNorm, error) {
	g, err := w.get(prefix+".weight", width)
	if err != nil {
		return layerNorm{}, err
	}
	b, err := w.get(prefix+".bias", width)
	return layerNorm{g, b}, err
}

func loadLinear(w weights, prefix string, in, out int, int8 bool) (*linear, error) {
	wt, err := w.get(prefix+".weight", out, in)
	if err != nil {
		return nil, err
	}
	b, err := w.get(prefix+".bias", out)
	if err != nil {
		return nil, err
	}
	return newLinear(wt, b, in, out, int8), nil
}

// layer is a pre-norm transformer layer:
// x += attn(ln1(x)); x += fc2(gelu(fc1(ln2(x)))).
type layer struct {
	ln1, ln2 layerNorm
	qkv, out *linear // q, k and v fused into one [3H, H] product
	fc1, fc2 *linear
}

// encoder is the stack of layers shared by both towers.
type encoder struct {
	cfg    Config
	layers []layer
}

func loadEncoder(cfg Config, w weights, prefix string, int8 bool) (*encoder, error) {
	H, M := cfg.Hidden, cfg.MLP
	e := &encoder{cfg: cfg, layers: make([]layer, cfg.Layers)}
	for i := range e.layers {
		p := fmt.Sprintf("%s.layers.%d.", prefix, i)
		l := &e.layers[i]
		var err error
		if l.ln1, err = loadNorm(w, p+"layer_norm1", H); err != nil {
			return nil, err
		}
		if l.ln2, err = loadNorm(w, p+"layer_norm2", H); err != nil {
			return nil, err
		}
		qkvW := make([]float32, 0, 3*H*H)
		qkvB := make([]float32, 0, 3*H)
		for _, name := range []string{"q_proj", "k_proj", "v_proj"} {
			wt, err := w.get(p+"self_attn."+name+".weight", H, H)
			if err != nil {
				return nil, err
			}
			b, err := w.get(p+"self_attn."+name+".bias", H)
			if err != nil {
				return nil, err
			}
			qkvW, qkvB = append(qkvW, wt...), append(qkvB, b...)
		}
		l.qkv = newLinear(qkvW, qkvB, H, 3*H, int8)
		if l.out, err = loadLinear(w, p+"self_attn.out_proj", H, H, int8); err != nil {
			return nil, err
		}
		if l.fc1, err = loadLinear(w, p+"mlp.fc1", H, M, int8); err != nil {
			return nil, err
		}
		// fc2 reads the GELU output, whose outliers (a known trait of vision
		// transformers) per-token int8 flattens: alone, it moved the logits
		// by up to 1.5 on the fixtures. It stays in float32.
		if l.fc2, err = loadLinear(w, p+"mlp.fc2", M, H, false); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// buffers holds the working memory of one forward pass over T rows.
type buffers struct {
	h, qkv, attn, tmp, mlp []float32
	q, k, v, s, o          []float32 // one head
}

func newBuffers(cfg Config, T int) *buffers {
	H, d := cfg.Hidden, cfg.headDim()
	return &buffers{
		h: make([]float32, T*H), qkv: make([]float32, T*3*H), attn: make([]float32, T*H),
		tmp: make([]float32, T*H), mlp: make([]float32, T*cfg.MLP),
		q: make([]float32, T*d), k: make([]float32, T*d), v: make([]float32, T*d),
		s: make([]float32, T*T), o: make([]float32, T*d),
	}
}

// forward runs the layers on x, [T, H], in place.
func (e *encoder) forward(x []float32, T int, b *buffers) {
	cfg := e.cfg
	H := cfg.Hidden
	for i := range e.layers {
		l := &e.layers[i]
		l.ln1.apply(b.h, x, T, H, cfg.Eps)
		l.qkv.apply(b.qkv, b.h, T)
		attention(b.attn, b.qkv, T, cfg, b)
		l.out.apply(b.tmp, b.attn, T)
		linalg.AddTo(x[:T*H], b.tmp[:T*H])

		l.ln2.apply(b.h, x, T, H, cfg.Eps)
		l.fc1.apply(b.mlp, b.h, T)
		linalg.GeluTanh(b.mlp[:T*cfg.MLP], b.mlp[:T*cfg.MLP])
		l.fc2.apply(b.tmp, b.mlp, T)
		linalg.AddTo(x[:T*H], b.tmp[:T*H])
	}
}

// attention computes the full (unmasked) multi-head attention of the
// fused q, k, v rows, [T, 3H], into dst, [T, H].
func attention(dst, qkv []float32, T int, cfg Config, b *buffers) {
	H, d := cfg.Hidden, cfg.headDim()
	scale := float32(1 / math.Sqrt(float64(d)))
	for h := 0; h < cfg.Heads; h++ {
		for t := 0; t < T; t++ {
			row := qkv[t*3*H:]
			copy(b.q[t*d:(t+1)*d], row[h*d:(h+1)*d])
			copy(b.k[t*d:(t+1)*d], row[H+h*d:H+(h+1)*d])
			copy(b.v[t*d:(t+1)*d], row[2*H+h*d:2*H+(h+1)*d])
		}
		linalg.MatMul(b.s[:T*T], b.q[:T*d], b.k[:T*d], T, d, T, false, true, false)
		for t := 0; t < T; t++ {
			softmax(b.s[t*T:(t+1)*T], scale)
		}
		linalg.MatMul(b.o[:T*d], b.s[:T*T], b.v[:T*d], T, T, d, false, false, false)
		for t := 0; t < T; t++ {
			copy(dst[t*H+h*d:t*H+(h+1)*d], b.o[t*d:(t+1)*d])
		}
	}
}

// softmax replaces the raw scores s by softmax(s·scale).
func softmax(s []float32, scale float32) {
	linalg.Scale(s, scale)
	sum := linalg.ExpShift(s, linalg.MaxOf(s))
	linalg.Scale(s, 1/sum)
}
