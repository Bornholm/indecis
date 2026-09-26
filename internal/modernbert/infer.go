package modernbert

import (
	"fmt"
	"math"
	"sync"

	"github.com/bornholm/indecis/internal/linalg"
)

// EmbeddingTable fournit les lignes de la table d'embeddings à la demande.
// La table fait 93 % des poids ; en inférence, une requête n'en lit que
// quelques dizaines de lignes, qu'il est inutile de garder en float32.
type EmbeddingTable interface {
	// Row écrit la ligne id dans dst (Hidden valeurs).
	Row(id int32, dst []float32)
}

// SetEmbeddingTable remplace la table d'embeddings en float32 par une table
// lue à la demande. Emb.W vaut alors nil jusqu'à Materialize.
func (m *Model) SetEmbeddingTable(t EmbeddingTable) {
	m.embTable = t
	m.Emb.W = nil
}

// Materialize recopie la table d'embeddings en float32 dans Emb.W, ce
// qu'exige l'entraînement.
func (m *Model) Materialize() {
	m.restoreWeights()
	if m.Emb.W != nil {
		return
	}
	m.Emb.W = m.EmbeddingMatrix()
	m.embTable = nil
}

// EmbeddingMatrix retourne la table d'embeddings en float32 : Emb.W, ou une
// copie temporaire si la table est lue à la demande.
func (m *Model) EmbeddingMatrix() []float32 {
	if m.Emb.W != nil {
		return m.Emb.W
	}
	H := m.Cfg.Hidden
	w := make([]float32, m.Cfg.Vocab*H)
	for id := 0; id < m.Cfg.Vocab; id++ {
		m.embTable.Row(int32(id), w[id*H:(id+1)*H])
	}
	return w
}

func (m *Model) embRow(id int32, dst []float32) {
	if m.Emb.W != nil {
		H := m.Cfg.Hidden
		copy(dst, m.Emb.W[int(id)*H:(int(id)+1)*H])
		return
	}
	m.embTable.Row(id, dst)
}

// packedMat est une matrice de poids prête pour le produit : empaquetée en
// float32, ou quantifiée en int8 (voir SetInt8).
type packedMat struct {
	f *linalg.PackedB
	q *linalg.PackedB8
}

func packMat(w []float32, k, n int, int8 bool) packedMat {
	if int8 {
		return packedMat{q: linalg.PackB8(w, k, n, true)}
	}
	return packedMat{f: linalg.PackB(w, k, n, true)}
}

// mul calcule c = a·Wᵀ (ou c += si accumulate), a ayant m lignes.
func (p packedMat) mul(c, a []float32, m int, accumulate bool) {
	if p.q != nil {
		linalg.MatMul8(c, a, p.q, m, accumulate)
		return
	}
	linalg.MatMulPacked(c, a, p.f, m, accumulate)
}

// packedLayer contient les matrices d'une couche prêtes pour le produit.
type packedLayer struct {
	qkv, o, i, oMLP packedMat
}

// WeightSource relit une matrice de poids par son nom, dans le fichier du
// modèle par exemple.
type WeightSource func(name string) ([]float32, error)

// Invalidate signale que les poids ont changé : les matrices empaquetées
// pour l'inférence sont à refaire. L'entraînement l'appelle après chaque
// pas.
func (m *Model) Invalidate() {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	m.restoreLocked()
	m.packed = nil
}

// SetCompact fait garder les matrices des couches sous leur seule forme
// empaquetée, dès la première inférence : elles ne sont plus en double. Les
// fonctions qui lisent les poids (Params, Forward, Materialize) les
// reconstruisent au besoin, depuis source si elle est fournie, et le mode
// compact prend fin. En int8, les matrices ne sont libérées que si source
// est fournie : la quantification ne se défait pas.
func (m *Model) SetCompact(source WeightSource) {
	m.packMu.Lock()
	m.compact = true
	m.source = source
	m.packMu.Unlock()
}

// SetInt8 fait calculer les produits par les couches en int8 (poids par
// canal, activations par token). Sans noyau matériel (linalg.Int8Fast),
// c'est exact mais lent.
func (m *Model) SetInt8(on bool) {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	if m.int8 == on {
		return
	}
	compact := m.compact
	m.restoreLocked()
	m.compact = compact // changer de format n'est pas lire les poids
	m.int8 = on
	m.packed = nil
}

// restoreWeights reconstruit les matrices des couches si SetCompact les a
// libérées.
func (m *Model) restoreWeights() {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	m.restoreLocked()
}

func (m *Model) restoreLocked() {
	m.compact = false
	for l := range m.Layers {
		L := &m.Layers[l]
		var p packedLayer
		if m.packed != nil {
			p = m.packed[l]
		}
		for _, x := range []struct {
			w *Param
			p packedMat
		}{{L.Wqkv, p.qkv}, {L.Wo, p.o}, {L.Wi, p.i}, {L.WoMLP, p.oMLP}} {
			if x.w.W != nil {
				continue
			}
			if m.source != nil {
				w, err := m.source(x.w.Name)
				if err == nil && len(w) != x.w.Shape[0]*x.w.Shape[1] {
					err = fmt.Errorf("%d valeurs", len(w))
				}
				if err != nil {
					panic(fmt.Sprintf("modernbert: relecture de %s : %v", x.w.Name, err))
				}
				x.w.W = w
				continue
			}
			x.w.W = make([]float32, x.w.Shape[0]*x.w.Shape[1])
			x.p.f.Unpack(x.w.W, true)
		}
	}
}

func (m *Model) packs() []packedLayer {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	if m.packed != nil {
		return m.packed
	}
	H, I := m.Cfg.Hidden, m.Cfg.Intermediate
	p := make([]packedLayer, len(m.Layers))
	for l, L := range m.Layers {
		p[l] = packedLayer{
			qkv:  packMat(L.Wqkv.W, H, 3*H, m.int8),
			o:    packMat(L.Wo.W, H, H, m.int8),
			i:    packMat(L.Wi.W, H, 2*I, m.int8),
			oMLP: packMat(L.WoMLP.W, I, H, m.int8),
		}
		if m.compact && (m.source != nil || !m.int8) {
			L.Wqkv.W, L.Wo.W, L.Wi.W, L.WoMLP.W = nil, nil, nil, nil
		}
	}
	m.packed = p
	return p
}

// workspace regroupe les tampons d'un passage d'inférence, réutilisés d'une
// requête à l'autre.
type workspace struct {
	x, xn, qkv, ctx, tmp, z, g []float32
}

var workspaces sync.Pool

// Encode calcule le plongement moyen de chaque séquence du lot, [B, H].
//
// C'est le chemin d'inférence : même calcul que Forward suivi de MeanPool,
// sans rien garder pour la rétropropagation, avec des poids empaquetés une
// fois pour toutes et des erf et exp en float32. L'écart avec Forward reste
// de l'ordre de 1e-6.
func (m *Model) Encode(b Batch) ([]float32, error) {
	cfg := m.Cfg
	H, I := cfg.Hidden, cfg.Intermediate
	N := b.B() * b.T
	if len(b.IDs) != N {
		return nil, fmt.Errorf("modernbert: lot incohérent")
	}
	for _, id := range b.IDs {
		if id < 0 || int(id) >= cfg.Vocab {
			return nil, fmt.Errorf("modernbert: id %d hors vocabulaire", id)
		}
	}
	packs := m.packs()

	ws, _ := workspaces.Get().(*workspace)
	if ws == nil {
		ws = &workspace{}
	}
	defer workspaces.Put(ws)
	ws.x = grow(ws.x, N*H)
	ws.xn = grow(ws.xn, N*H)
	ws.qkv = grow(ws.qkv, N*3*H)
	ws.ctx = grow(ws.ctx, N*H)
	ws.tmp = grow(ws.tmp, N*H)
	ws.z = grow(ws.z, N*2*I)
	ws.g = grow(ws.g, N*I)
	x, xn := ws.x, ws.xn

	for r, id := range b.IDs {
		m.embRow(id, xn[r*H:(r+1)*H])
	}
	layerNormInfer(x, xn, m.EmbNorm.W, N, H, cfg.NormEps)

	for l, L := range m.Layers {
		p := packs[l]
		attnIn := x
		if L.AttnNorm != nil {
			layerNormInfer(xn, x, L.AttnNorm.W, N, H, cfg.NormEps)
			attnIn = xn
		}
		p.qkv.mul(ws.qkv, attnIn, N, false)
		m.attentionInfer(l, b, ws.qkv, ws.ctx)
		p.o.mul(x, ws.ctx, N, true) // x += attention

		layerNormInfer(xn, x, L.MLPNorm.W, N, H, cfg.NormEps)
		p.i.mul(ws.z, xn, N, false)
		gluInfer(ws.g, ws.z, N, I)
		p.oMLP.mul(x, ws.g, N, true) // x += MLP
	}
	layerNormInfer(xn, x, m.FinalNorm.W, N, H, cfg.NormEps)
	return MeanPool(xn, b, H), nil
}

// attentionInfer est attentionForward sans rien conserver : q, k, v et les
// scores de chaque (séquence, tête) vivent dans des tampons de la tâche.
func (m *Model) attentionInfer(l int, b Batch, qkv, ctx []float32) {
	cfg := m.Cfg
	H, nh, D := cfg.Hidden, cfg.Heads, cfg.HeadDim()
	T := b.T
	half := D / 2
	rope := m.ropeFor(m.theta(l), T)
	scale := float32(1 / math.Sqrt(float64(D)))
	window := -1
	if !cfg.IsGlobal(l) {
		window = cfg.Window()
	}
	linalg.Parallel(b.B()*nh, 1, func(lo, hi int) {
		buf := make([]float32, 4*T*D+T*T)
		q, k, v, o := buf[:T*D], buf[T*D:2*T*D], buf[2*T*D:3*T*D], buf[3*T*D:4*T*D]
		s := buf[4*T*D:]
		for task := lo; task < hi; task++ {
			bi, h := task/nh, task%nh
			n := b.Lens[bi]
			if n == 0 {
				continue
			}
			for t := 0; t < n; t++ {
				row := qkv[(bi*T+t)*3*H:]
				cos := rope.cos[t*half : (t+1)*half]
				sin := rope.sin[t*half : (t+1)*half]
				qt, kt := q[t*D:(t+1)*D], k[t*D:(t+1)*D]
				copy(qt, row[h*D:(h+1)*D])
				copy(kt, row[H+h*D:H+(h+1)*D])
				copy(v[t*D:(t+1)*D], row[2*H+h*D:2*H+(h+1)*D])
				applyRope(qt, cos, sin)
				applyRope(kt, cos, sin)
			}
			p := s[:n*n]
			linalg.MatMulSerial(p, q, k, n, D, n, false, true, false)
			for i := 0; i < n; i++ {
				row := p[i*n : (i+1)*n]
				j0, j1 := 0, n
				if window >= 0 {
					j0, j1 = max(0, i-window), min(n, i+window+1)
				}
				mx := float32(math.Inf(-1))
				for j := j0; j < j1; j++ {
					row[j] *= scale
					mx = max(mx, row[j])
				}
				var sum float32
				for j := j0; j < j1; j++ {
					e := exp32(row[j] - mx)
					row[j] = e
					sum += e
				}
				inv := 1 / sum
				clear(row[:j0])
				clear(row[j1:])
				for j := j0; j < j1; j++ {
					row[j] *= inv
				}
			}
			linalg.MatMulSerial(o, p, v, n, n, D, false, false, false)
			for t := 0; t < n; t++ {
				copy(ctx[(bi*T+t)*H+h*D:(bi*T+t)*H+(h+1)*D], o[t*D:(t+1)*D])
			}
		}
	})
	// Les positions de padding ne sont lues par personne, mais le produit
	// par Wo les additionne au flux résiduel : on les garde à zéro.
	for bi, n := range b.Lens {
		clear(ctx[(bi*T+n)*H : (bi+1)*T*H])
	}
}

// layerNormInfer est layerNorm sans cache pour le backward.
func layerNormInfer(out, x, gamma []float32, n, h int, eps float64) {
	linalg.Parallel(n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x[r*h : (r+1)*h]
			var mean float64
			for _, v := range row {
				mean += float64(v)
			}
			mean /= float64(h)
			var variance float64
			for _, v := range row {
				d := float64(v) - mean
				variance += d * d
			}
			variance /= float64(h)
			rstd := float32(1 / math.Sqrt(variance+eps))
			mf := float32(mean)
			o := out[r*h : (r+1)*h]
			for i, v := range row {
				o[i] = (v - mf) * rstd * gamma[i]
			}
		}
	})
}

// gluInfer est gluForward avec la GELU en float32.
func gluInfer(g, z []float32, n, inter int) {
	linalg.Parallel(n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			a := z[r*2*inter : r*2*inter+inter]
			b := z[r*2*inter+inter : (r+1)*2*inter]
			o := g[r*inter : (r+1)*inter]
			for i := range o {
				o[i] = geluFast(a[i]) * b[i]
			}
		}
	})
}
