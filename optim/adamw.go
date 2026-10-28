// Package optim implements AdamW, identical to torch.optim.AdamW, and its
// sparse variant for embedding tables.
package optim

import "math"

// Dense is a dense parameter and its gradient.
type Dense struct {
	W, G []float32
	// Decay applies decoupled weight decay. It is customary to exclude it
	// from normalizations and embeddings.
	Decay bool
}

// Sparse is a table where only some rows have a gradient.
type Sparse struct {
	W     []float32
	Width int
	Rows  map[int32][]float32 // gradient per row
}

// Config mirrors the hyperparameters of torch.optim.AdamW.
type Config struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
}

// DefaultConfig is PyTorch's default configuration, with the weight decay
// usual for fine-tuning.
func DefaultConfig(lr float64) Config {
	return Config{LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.01}
}

// AdamW keeps the moments of each parameter. The parameters must be passed
// in the same order at each Step.
type AdamW struct {
	Cfg  Config
	step int
	m, v [][]float32
	// Moments of embedding rows already seen: "lazy" Adam, as in
	// torch.optim.SparseAdam. A row absent from the batch keeps its moments
	// and its weights.
	sm, sv []map[int32][]float32
}

// New creates an optimizer.
func New(cfg Config) *AdamW { return &AdamW{Cfg: cfg} }

// Steps returns the number of steps taken.
func (o *AdamW) Steps() int { return o.step }

// Step applies a step to all parameters, at rate lr (which overrides
// Cfg.LR, for rate schedules).
func (o *AdamW) Step(lr float64, dense []Dense, sparse []Sparse) {
	if o.m == nil {
		o.m = make([][]float32, len(dense))
		o.v = make([][]float32, len(dense))
		for i, p := range dense {
			o.m[i] = make([]float32, len(p.W))
			o.v[i] = make([]float32, len(p.W))
		}
		o.sm = make([]map[int32][]float32, len(sparse))
		o.sv = make([]map[int32][]float32, len(sparse))
		for i := range sparse {
			o.sm[i] = map[int32][]float32{}
			o.sv[i] = map[int32][]float32{}
		}
	}
	o.step++
	c := o.Cfg
	bc1 := 1 - math.Pow(c.Beta1, float64(o.step))
	bc2 := 1 - math.Pow(c.Beta2, float64(o.step))
	stepSize := lr / bc1
	sqrtBC2 := math.Sqrt(bc2)

	for i, p := range dense {
		decay := 1.0
		if p.Decay {
			decay = 1 - lr*c.WeightDecay
		}
		update(p.W, p.G, o.m[i], o.v[i], decay, stepSize, sqrtBC2, c)
	}
	for i, p := range sparse {
		for id, g := range p.Rows {
			m, ok := o.sm[i][id]
			if !ok {
				m = make([]float32, p.Width)
				o.sm[i][id] = m
				o.sv[i][id] = make([]float32, p.Width)
			}
			w := p.W[int(id)*p.Width : (int(id)+1)*p.Width]
			update(w, g, m, o.sv[i][id], 1, stepSize, sqrtBC2, c)
		}
	}
}

// update follows the operation order of PyTorch's "for-loop"
// implementation, so that rounding matches.
func update(w, g, m, v []float32, decay, stepSize, sqrtBC2 float64, c Config) {
	b1, b2 := float32(c.Beta1), float32(c.Beta2)
	for j := range w {
		gj := g[j]
		if decay != 1 {
			w[j] = float32(float64(w[j]) * decay)
		}
		m[j] = m[j]*b1 + (1-b1)*gj
		v[j] = v[j]*b2 + (1-b2)*gj*gj
		denom := math.Sqrt(float64(v[j]))/sqrtBC2 + c.Eps
		w[j] = float32(float64(w[j]) - stepSize*float64(m[j])/denom)
	}
}

// ClipGradNorm scales the global L2 norm of the gradients down to maxNorm if
// it exceeds it, and returns the norm before clipping.
func ClipGradNorm(maxNorm float64, dense []Dense, sparse []Sparse) float64 {
	var sq float64
	for _, p := range dense {
		for _, g := range p.G {
			sq += float64(g) * float64(g)
		}
	}
	for _, p := range sparse {
		for _, row := range p.Rows {
			for _, g := range row {
				sq += float64(g) * float64(g)
			}
		}
	}
	norm := math.Sqrt(sq)
	if maxNorm <= 0 || norm <= maxNorm {
		return norm
	}
	coef := float32(maxNorm / (norm + 1e-6))
	for _, p := range dense {
		for j := range p.G {
			p.G[j] *= coef
		}
	}
	for _, p := range sparse {
		for _, row := range p.Rows {
			for j := range row {
				row[j] *= coef
			}
		}
	}
	return norm
}
