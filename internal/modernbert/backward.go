package modernbert

import (
	"sort"

	"github.com/bornholm/indecis/internal/linalg"
)

// SparseGrad est le gradient d'une table d'embeddings : seules les lignes
// des tokens présents dans le lot sont non nulles. Garder un gradient dense
// de 256 000 × 384 valeurs pour en toucher quelques centaines coûterait
// 400 Mo par copie.
type SparseGrad struct {
	Width int
	Rows  map[int32][]float32
}

// IDs retourne les lignes non nulles, triées.
func (g *SparseGrad) IDs() []int32 {
	ids := make([]int32, 0, len(g.Rows))
	for id := range g.Rows {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// Grads accumule les gradients de tous les paramètres.
type Grads struct {
	Emb SparseGrad
}

// EnableGrad alloue les gradients des paramètres denses et prépare le
// gradient creux des embeddings.
func (m *Model) EnableGrad() *Grads {
	for _, p := range m.Params() {
		if p == m.Emb {
			continue
		}
		if p.G == nil {
			p.G = make([]float32, len(p.W))
		}
	}
	return &Grads{Emb: SparseGrad{Width: m.Cfg.Hidden, Rows: map[int32][]float32{}}}
}

// ZeroGrad remet les gradients à zéro.
func (m *Model) ZeroGrad(g *Grads) {
	for _, p := range m.Params() {
		clear(p.G)
	}
	clear(g.Emb.Rows)
}

// Backward rétropropage dHidden (gradient de la perte par rapport à
// State.Hidden, [N, H]) et accumule les gradients des paramètres.
func (m *Model) Backward(s *State, dHidden []float32, g *Grads) {
	cfg := m.Cfg
	H := cfg.Hidden
	N := s.N

	dx := make([]float32, N*H)
	layerNormBackward(dx, dHidden, m.FinalNorm.W, m.FinalNorm.G, N, H, &s.final)

	for l := cfg.Layers - 1; l >= 0; l-- {
		dx = m.layerBackward(l, s, dx)
	}

	dEmb := make([]float32, N*H)
	layerNormBackward(dEmb, dx, m.EmbNorm.W, m.EmbNorm.G, N, H, &s.embLN)

	b := s.Batch
	for bi, n := range b.Lens {
		for t := 0; t < n; t++ {
			r := bi*b.T + t
			id := b.IDs[r]
			row := g.Emb.Rows[id]
			if row == nil {
				row = make([]float32, H)
				g.Emb.Rows[id] = row
			}
			for i, v := range dEmb[r*H : (r+1)*H] {
				row[i] += v
			}
		}
	}
}

// layerBackward retourne le gradient par rapport à l'entrée de la couche.
func (m *Model) layerBackward(l int, s *State, dOut []float32) []float32 {
	cfg := m.Cfg
	L := m.Layers[l]
	ls := &s.layers[l]
	H, I := cfg.Hidden, cfg.Intermediate
	N := s.N

	// out = h1 + WoMLP(g)
	dh1 := append([]float32(nil), dOut...)
	dg := make([]float32, N*I)
	linalg.MatMul(dg, dOut, L.WoMLP.W, N, H, I, false, false, false)
	linalg.MatMul(L.WoMLP.G, dOut, ls.g, H, N, I, true, false, true)

	// g = gelu(a) ⊙ porte, z = mlpIn · Wiᵀ
	dz := make([]float32, N*2*I)
	gluBackward(dz, dg, ls.z, N, I)
	dMLPIn := make([]float32, N*H)
	linalg.MatMul(dMLPIn, dz, L.Wi.W, N, 2*I, H, false, false, false)
	linalg.MatMul(L.Wi.G, dz, ls.mlpIn, 2*I, N, H, true, false, true)

	// mlpIn = mlp_norm(h1)
	layerNormBackward(dh1, dMLPIn, L.MLPNorm.W, L.MLPNorm.G, N, H, &ls.mlpLN)

	// h1 = in + Wo(ctx)
	dIn := append([]float32(nil), dh1...)
	dCtx := make([]float32, N*H)
	linalg.MatMul(dCtx, dh1, L.Wo.W, N, H, H, false, false, false)
	linalg.MatMul(L.Wo.G, dh1, ls.ctx, H, N, H, true, false, true)

	dqkv := make([]float32, N*3*H)
	m.attentionBackward(l, s, dCtx, dqkv)

	// qkv = attnIn · Wqkvᵀ
	dAttnIn := make([]float32, N*H)
	linalg.MatMul(dAttnIn, dqkv, L.Wqkv.W, N, 3*H, H, false, false, false)
	linalg.MatMul(L.Wqkv.G, dqkv, ls.attnIn, 3*H, N, H, true, false, true)

	if L.AttnNorm == nil {
		for i, v := range dAttnIn {
			dIn[i] += v
		}
	} else {
		layerNormBackward(dIn, dAttnIn, L.AttnNorm.W, L.AttnNorm.G, N, H, &ls.attnLN)
	}
	return dIn
}

// attentionBackward calcule le gradient par rapport à qkv à partir du
// gradient des têtes concaténées.
func (m *Model) attentionBackward(l int, s *State, dCtx, dqkv []float32) {
	cfg := m.Cfg
	H, nh, D := cfg.Hidden, cfg.Heads, cfg.HeadDim()
	b := s.Batch
	B, T := b.B(), b.T
	half := D / 2
	ls := &s.layers[l]
	rope := m.ropeFor(m.theta(l), T)
	scale := float32(1 / sqrt(float64(D)))

	linalg.Parallel(B*nh, 1, func(lo, hi int) {
		dO := make([]float32, T*D)
		dV := make([]float32, T*D)
		dQ := make([]float32, T*D)
		dK := make([]float32, T*D)
		dP := make([]float32, T*T)
		for task := lo; task < hi; task++ {
			bi, h := task/nh, task%nh
			n := b.Lens[bi]
			if n == 0 {
				continue
			}
			base := (bi*nh + h) * T * D
			q := ls.q[base : base+n*D]
			k := ls.k[base : base+n*D]
			v := ls.v[base : base+n*D]
			p := ls.p[(bi*nh+h)*T*T : (bi*nh+h)*T*T+n*n]
			for t := 0; t < n; t++ {
				copy(dO[t*D:(t+1)*D], dCtx[(bi*T+t)*H+h*D:(bi*T+t)*H+(h+1)*D])
			}

			// O = P·V
			linalg.MatMulSerial(dV, p, dO, n, n, D, true, false, false)
			linalg.MatMulSerial(dP, dO, v, n, D, n, false, true, false)

			// P = softmax(S) : dS = P ⊙ (dP - Σ_j dP·P), réécrit dans dP.
			for i := 0; i < n; i++ {
				pr := p[i*n : (i+1)*n]
				dr := dP[i*n : (i+1)*n]
				var dot float64
				for j := range pr {
					dot += float64(pr[j]) * float64(dr[j])
				}
				for j := range pr {
					dr[j] = pr[j] * (dr[j] - float32(dot)) * scale
				}
			}

			// S = scale · Q·Kᵀ (le facteur scale est déjà dans dS)
			linalg.MatMulSerial(dQ, dP, k, n, n, D, false, false, false)
			linalg.MatMulSerial(dK, dP, q, n, n, D, true, false, false)

			for t := 0; t < n; t++ {
				cos := rope.cos[t*half : (t+1)*half]
				sin := rope.sin[t*half : (t+1)*half]
				applyRopeBackward(dQ[t*D:(t+1)*D], cos, sin)
				applyRopeBackward(dK[t*D:(t+1)*D], cos, sin)
				row := dqkv[(bi*T+t)*3*H : (bi*T+t+1)*3*H]
				copy(row[h*D:(h+1)*D], dQ[t*D:(t+1)*D])
				copy(row[H+h*D:H+(h+1)*D], dK[t*D:(t+1)*D])
				copy(row[2*H+h*D:2*H+(h+1)*D], dV[t*D:(t+1)*D])
			}
		}
	})
}

// MeanPoolBackward répartit le gradient de chaque vecteur poolé sur les
// tokens réels de sa séquence.
func MeanPoolBackward(dPooled []float32, b Batch, h int) []float32 {
	d := make([]float32, b.B()*b.T*h)
	for bi, n := range b.Lens {
		if n == 0 {
			continue
		}
		inv := 1 / float32(n)
		src := dPooled[bi*h : (bi+1)*h]
		for t := 0; t < n; t++ {
			row := d[(bi*b.T+t)*h : (bi*b.T+t+1)*h]
			for i, v := range src {
				row[i] = v * inv
			}
		}
	}
	return d
}
