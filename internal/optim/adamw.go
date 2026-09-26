// Package optim implémente AdamW, à l'identique de torch.optim.AdamW, et sa
// variante creuse pour les tables d'embeddings.
package optim

import "math"

// Dense est un paramètre dense et son gradient.
type Dense struct {
	W, G []float32
	// Decay applique le weight decay découplé. Il est d'usage de l'exclure
	// des normalisations et des embeddings.
	Decay bool
}

// Sparse est une table dont seules certaines lignes ont un gradient.
type Sparse struct {
	W     []float32
	Width int
	Rows  map[int32][]float32 // gradient par ligne
}

// Config reprend les hyperparamètres de torch.optim.AdamW.
type Config struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
}

// DefaultConfig est la configuration par défaut de PyTorch, avec le weight
// decay usuel du fine-tuning.
func DefaultConfig(lr float64) Config {
	return Config{LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.01}
}

// AdamW garde les moments de chaque paramètre. Les paramètres doivent être
// passés dans le même ordre à chaque Step.
type AdamW struct {
	Cfg  Config
	step int
	m, v [][]float32
	// Moments des lignes d'embedding déjà vues : Adam « paresseux », comme
	// torch.optim.SparseAdam. Une ligne absente du lot garde ses moments et
	// ses poids.
	sm, sv []map[int32][]float32
}

// New crée un optimiseur.
func New(cfg Config) *AdamW { return &AdamW{Cfg: cfg} }

// Steps retourne le nombre de pas effectués.
func (o *AdamW) Steps() int { return o.step }

// Step applique un pas à tous les paramètres, au taux lr (qui remplace
// Cfg.LR, pour les calendriers de taux).
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

// update suit l'ordre des opérations de l'implémentation « for-loop » de
// PyTorch, pour que les arrondis soient les mêmes.
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

// ClipGradNorm ramène la norme L2 globale des gradients à maxNorm si elle la
// dépasse, et retourne la norme avant écrêtage.
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
