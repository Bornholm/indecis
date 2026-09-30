package siglip

import (
	"fmt"
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// Vision is the image tower: patch embedding, transformer layers, a final
// LayerNorm and multi-head attention pooling.
type Vision struct {
	cfg   Config
	work  *workPool
	patch *linear   // the patch convolution, as a product over flattened patches
	pos   []float32 // [patches, H]
	enc   *encoder
	post  layerNorm

	// Attention pooling: a learned query (probe) attends to the patches,
	// then a residual MLP.
	probeQ           []float32 // probe·Wqᵀ + bq, computed once
	headK, headV     *linear
	headOut          *linear
	headNorm         layerNorm
	headFC1, headFC2 *linear
}

func loadVision(cfg Config, w weights, quantized bool) (*Vision, error) {
	H, P := cfg.Hidden, cfg.PatchSize
	v := &Vision{cfg: cfg, work: &workPool{cfg: cfg, T: cfg.Patches(), rows: cfg.Patches() * 3 * P * P}}
	var err error
	// Conv2d weight [H, 3, P, P]: a row per output channel, flattened in
	// (channel, y, x) order, the order patchify uses.
	// The patch embedding and the pooling head stay in float32: they are a
	// small share of the compute, and int8 there costs precision.
	if v.patch, err = loadLinear(w, "vision_model.embeddings.patch_embedding", 3*P*P, H, false); err != nil {
		return nil, err
	}
	if v.pos, err = w.get("vision_model.embeddings.position_embedding.weight", cfg.Patches(), H); err != nil {
		return nil, err
	}
	if v.enc, err = loadEncoder(cfg, w, "vision_model.encoder", quantized, true); err != nil {
		return nil, err
	}
	if v.post, err = loadNorm(w, "vision_model.post_layernorm", H); err != nil {
		return nil, err
	}
	probe, err := w.get("vision_model.head.probe", 1, 1, H)
	if err != nil {
		return nil, err
	}
	inW, err := w.get("vision_model.head.attention.in_proj_weight", 3*H, H)
	if err != nil {
		return nil, err
	}
	inB, err := w.get("vision_model.head.attention.in_proj_bias", 3*H)
	if err != nil {
		return nil, err
	}
	// The query only depends on the probe: computed once, in float32.
	q := newLinear(inW[:H*H], inB[:H], H, H, false)
	v.probeQ = make([]float32, H)
	q.apply(v.probeQ, probe, 1, 1)
	v.headK = newLinear(inW[H*H:2*H*H], inB[H:2*H], H, H, false)
	v.headV = newLinear(inW[2*H*H:], inB[2*H:], H, H, false)
	if v.headOut, err = loadLinear(w, "vision_model.head.attention.out_proj", H, H, false); err != nil {
		return nil, err
	}
	if v.headNorm, err = loadNorm(w, "vision_model.head.layernorm", H); err != nil {
		return nil, err
	}
	if v.headFC1, err = loadLinear(w, "vision_model.head.mlp.fc1", H, cfg.MLP, false); err != nil {
		return nil, err
	}
	if v.headFC2, err = loadLinear(w, "vision_model.head.mlp.fc2", cfg.MLP, H, false); err != nil {
		return nil, err
	}
	return v, nil
}

// Embed computes the embedding of an image given as normalized pixels,
// [3, S, S] channel-major with S = ImageSize (see Pixels). The result,
// [H], is not normalized.
func (v *Vision) Embed(pixels []float32) ([]float32, error) {
	_, pooled, err := v.forward(pixels, 0, false, true)
	return pooled, err
}

// Forward runs the image tower on normalized pixels. layer > 0 returns the
// patch features after that many layers (hidden states, before the final
// LayerNorm), 0 the final ones; without pool, the layers beyond are not
// computed. With pool, it also returns the pooled embedding.
func (v *Vision) Forward(pixels []float32, layer int, pool bool) (patches, pooled []float32, err error) {
	return v.forward(pixels, layer, true, pool)
}

func (v *Vision) forward(pixels []float32, layer int, wantPatches, pool bool) (patches, pooled []float32, err error) {
	cfg := v.cfg
	S, P, H := cfg.ImageSize, cfg.PatchSize, cfg.Hidden
	if len(pixels) != 3*S*S {
		return nil, nil, fmt.Errorf("siglip: %d pixel values, expected 3×%d×%d", len(pixels), S, S)
	}
	if layer < 0 || layer > cfg.Layers {
		return nil, nil, fmt.Errorf("siglip: layer %d, the tower has %d", layer, cfg.Layers)
	}
	T := cfg.Patches()
	w := v.work.get()
	defer v.work.put(w)
	x, b := w.x, w.b
	patchify(w.rows, pixels, S, P)
	v.patch.apply(x, w.rows, T, v.enc.workers)
	linalg.AddTo(x, v.pos)
	n := cfg.Layers
	if layer > 0 && !pool {
		n = layer
	}
	if layer > 0 && wantPatches {
		patches = make([]float32, T*H)
	}
	v.enc.forwardN(x, T, b, n, layer, patches)
	if n < cfg.Layers {
		return patches, nil, nil
	}
	v.post.apply(x, x, T, H, cfg.Eps)
	if layer == 0 && wantPatches {
		patches = append([]float32(nil), x...) // x goes back to the pool
	}
	if pool {
		pooled = v.pool(x, T, b)
	}
	return patches, pooled, nil
}

// EmbedPatches returns both the patch features and the pooled embedding,
// with one pass of the encoder.
func (v *Vision) EmbedPatches(pixels []float32) (patches, pooled []float32, err error) {
	return v.Forward(pixels, 0, true)
}

// Patches returns the features of each patch, [Patches, H] in raster
// order, after the final LayerNorm and before pooling: where the pooled
// embedding summarizes the image, they keep the position of what they see.
func (v *Vision) Patches(pixels []float32) ([]float32, error) {
	x, _, err := v.Forward(pixels, 0, false)
	return x, err
}

// patchify cuts [3, S, S] pixels into P×P patches, one row per patch in
// raster order, each flattened in (channel, y, x) order.
func patchify(rows, pixels []float32, S, P int) {
	g := S / P
	i := 0
	for py := 0; py < g; py++ {
		for px := 0; px < g; px++ {
			for c := 0; c < 3; c++ {
				for y := 0; y < P; y++ {
					src := pixels[c*S*S+(py*P+y)*S+px*P:]
					copy(rows[i:i+P], src[:P])
					i += P
				}
			}
		}
	}
}

// pool is the attention pooling head: one query, the probe, attends to the
// T patch rows of x.
func (v *Vision) pool(x []float32, T int, b *buffers) []float32 {
	cfg := v.cfg
	H, d := cfg.Hidden, cfg.headDim()
	k, vv := b.qkv[:T*H], b.tmp[:T*H]
	v.headK.apply(k, x, T, v.enc.workers)
	v.headV.apply(vv, x, T, v.enc.workers)
	scale := float32(1 / math.Sqrt(float64(d)))
	att := make([]float32, H)
	s := make([]float32, T)
	for h := 0; h < cfg.Heads; h++ {
		q := v.probeQ[h*d : (h+1)*d]
		for t := 0; t < T; t++ {
			var dot float32
			kt := k[t*H+h*d : t*H+(h+1)*d]
			for j, qj := range q {
				dot += qj * kt[j]
			}
			s[t] = dot
		}
		softmax(s, scale)
		o := att[h*d : (h+1)*d]
		for t := 0; t < T; t++ {
			vt := vv[t*H+h*d : t*H+(h+1)*d]
			for j := range o {
				o[j] += s[t] * vt[j]
			}
		}
	}
	out := make([]float32, H)
	v.headOut.apply(out, att, 1, 1)
	// out + mlp(layernorm(out))
	n := make([]float32, H)
	v.headNorm.apply(n, out, 1, H, cfg.Eps)
	m := make([]float32, cfg.MLP)
	v.headFC1.apply(m, n, 1, 1)
	linalg.GeluTanh(m, m)
	v.headFC2.apply(n, m, 1, 1)
	linalg.AddTo(out, n)
	return out
}
