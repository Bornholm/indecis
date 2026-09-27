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
func (p packedMat) mul(c, a []float32, m int, accumulate bool, workers int) {
	if p.q != nil {
		linalg.MatMul8N(c, a, p.q, m, accumulate, workers)
		return
	}
	linalg.MatMulPackedN(c, a, p.f, m, accumulate, workers)
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
	x, xn, qkv, ctx, z, g []float32
}

var workspaces sync.Pool

// Encode calcule le plongement moyen de chaque séquence du lot, [B, H].
//
// C'est le chemin d'inférence : même calcul que Forward suivi de MeanPool,
// sans rien garder pour la rétropropagation, avec des poids empaquetés une
// fois pour toutes et des erf et exp en float32. L'écart avec Forward reste
// de l'ordre de 1e-6.
func (m *Model) Encode(b Batch) ([]float32, error) {
	if len(b.IDs) != b.B()*b.T {
		return nil, fmt.Errorf("modernbert: lot incohérent")
	}
	seqs := make([][]int32, b.B())
	for i, n := range b.Lens {
		seqs[i] = b.IDs[i*b.T : i*b.T+n]
	}
	return m.EncodeSeqs(seqs)
}

// EncodeSeqs est Encode sur des séquences de longueurs quelconques, mises
// bout à bout sans padding : les calculs par ligne (projections, MLP,
// normalisations) portent sur la somme des longueurs, pas sur le nombre
// de séquences fois la plus longue, et l'attention se calcule séquence par
// séquence. Réunir des séquences de longueurs différentes ne coûte donc
// rien de plus que de les calculer séparément.
func (m *Model) EncodeSeqs(seqs [][]int32) ([]float32, error) {
	cfg := m.Cfg
	H, I := cfg.Hidden, cfg.Intermediate
	offs := make([]int, len(seqs)+1)
	for i, sq := range seqs {
		for _, id := range sq {
			if id < 0 || int(id) >= cfg.Vocab {
				return nil, fmt.Errorf("modernbert: id %d hors vocabulaire", id)
			}
		}
		offs[i+1] = offs[i] + len(sq)
	}
	N := offs[len(seqs)]
	out := make([]float32, len(seqs)*H)
	if N == 0 {
		return out, nil
	}
	packs := m.packs()
	w := 1
	if N >= parallelRows {
		w = linalg.Workers()
	}

	ws, _ := workspaces.Get().(*workspace)
	if ws == nil {
		ws = &workspace{}
	}
	defer workspaces.Put(ws)
	ws.x = grow(ws.x, N*H)
	ws.xn = grow(ws.xn, N*H)
	ws.qkv = grow(ws.qkv, N*3*H)
	ws.ctx = grow(ws.ctx, N*H)
	ws.z = grow(ws.z, N*2*I)
	ws.g = grow(ws.g, N*I)
	x, xn := ws.x, ws.xn

	r := 0
	for _, sq := range seqs {
		for _, id := range sq {
			m.embRow(id, xn[r*H:(r+1)*H])
			r++
		}
	}
	layerNormInfer(x, xn, m.EmbNorm.W, N, H, cfg.NormEps, w)

	for l, L := range m.Layers {
		p := packs[l]
		attnIn := x
		if L.AttnNorm != nil {
			layerNormInfer(xn, x, L.AttnNorm.W, N, H, cfg.NormEps, w)
			attnIn = xn
		}
		p.qkv.mul(ws.qkv, attnIn, N, false, w)
		m.attentionInfer(l, offs, ws.qkv, ws.ctx, w)
		p.o.mul(x, ws.ctx, N, true, w) // x += attention

		layerNormInfer(xn, x, L.MLPNorm.W, N, H, cfg.NormEps, w)
		p.i.mul(ws.z, xn, N, false, w)
		gluInfer(ws.g, ws.z, N, I, w)
		p.oMLP.mul(x, ws.g, N, true, w) // x += MLP
	}
	layerNormInfer(xn, x, m.FinalNorm.W, N, H, cfg.NormEps, w)

	// Moyenne des états de chaque séquence, spéciaux compris.
	for i := range seqs {
		n := offs[i+1] - offs[i]
		if n == 0 {
			continue
		}
		acc := make([]float64, H)
		for t := offs[i]; t < offs[i+1]; t++ {
			for k, v := range xn[t*H : (t+1)*H] {
				acc[k] += float64(v)
			}
		}
		for k := range acc {
			out[i*H+k] = float32(acc[k] / float64(n))
		}
	}
	return out, nil
}

// attentionInfer est attentionForward sans rien conserver : q, k, v de
// chaque (séquence, tête) vivent dans des tampons de la tâche, et
// l'attention se calcule par blocs (attendBlocks), en mémoire linéaire.
func (m *Model) attentionInfer(l int, offs []int, qkv, ctx []float32, workers int) {
	cfg := m.Cfg
	H, nh, D := cfg.Hidden, cfg.Heads, cfg.HeadDim()
	B := len(offs) - 1
	T := 0
	for i := 0; i < B; i++ {
		T = max(T, offs[i+1]-offs[i])
	}
	half := D / 2
	rope := m.ropeFor(m.theta(l), T)
	scale := float32(1 / math.Sqrt(float64(D)))
	window := -1
	if !cfg.IsGlobal(l) {
		window = cfg.Window()
	}
	linalg.ParallelN(workers, B*nh, 1, func(lo, hi int) {
		buf := make([]float32, 4*T*D)
		q, k, v, o := buf[:T*D], buf[T*D:2*T*D], buf[2*T*D:3*T*D], buf[3*T*D:4*T*D]
		blk := newBlockBuffers()
		for task := lo; task < hi; task++ {
			bi, h := task/nh, task%nh
			off, n := offs[bi], offs[bi+1]-offs[bi]
			if n == 0 {
				continue
			}
			for t := 0; t < n; t++ {
				row := qkv[(off+t)*3*H:]
				cos := rope.cos[t*half : (t+1)*half]
				sin := rope.sin[t*half : (t+1)*half]
				qt, kt := q[t*D:(t+1)*D], k[t*D:(t+1)*D]
				copy(qt, row[h*D:(h+1)*D])
				copy(kt, row[H+h*D:H+(h+1)*D])
				copy(v[t*D:(t+1)*D], row[2*H+h*D:2*H+(h+1)*D])
				applyRope(qt, cos, sin)
				applyRope(kt, cos, sin)
			}
			attendBlocks(o, q, k, v, n, D, window, scale, blk)
			for t := 0; t < n; t++ {
				copy(ctx[(off+t)*H+h*D:(off+t)*H+(h+1)*D], o[t*D:(t+1)*D])
			}
		}
	})
}

// layerNormInfer est layerNorm sans cache pour le backward.
func layerNormInfer(out, x, gamma []float32, n, h int, eps float64, workers int) {
	linalg.ParallelN(workers, n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			linalg.LayerNormRow(out[r*h:(r+1)*h], x[r*h:(r+1)*h], gamma, eps)
		}
	})
}

// gluInfer est gluForward avec la GELU en float32.
func gluInfer(g, z []float32, n, inter int, workers int) {
	linalg.ParallelN(workers, n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			linalg.GeluMul(g[r*inter:(r+1)*inter], z[r*2*inter:r*2*inter+inter], z[r*2*inter+inter:(r+1)*2*inter])
		}
	})
}

// Attention par blocs, à la manière de FlashAttention : les requêtes sont
// traitées par blocs de qBlock positions, les clés par blocs de kBlock, et
// le softmax est mis à jour au fil des blocs de clés (maximum et somme
// courants par ligne). La matrice complète des scores n'existe jamais : la
// mémoire est linéaire en la longueur. Dans une couche locale, seuls les
// blocs de clés qui touchent la fenêtre sont parcourus : le calcul devient
// linéaire lui aussi. Le résultat est celui du softmax complet, à
// l'arrondi près.
const (
	qBlock = 64
	kBlock = 128

	// parallelRows est le nombre de positions (toutes séquences du lot
	// confondues) à partir duquel Encode répartit son calcul sur plusieurs
	// cœurs. En deçà, lancer des goroutines coûte plus que le calcul, et un
	// processeur hybride les envoie sur ses cœurs lents : une phrase se
	// calcule deux fois plus vite sur un seul cœur.
	parallelRows = 1024
)

type blockBuffers struct {
	s       []float32 // scores d'un bloc, [qBlock, kBlock]
	mx, sum []float32 // maximum et somme courants par ligne
}

func newBlockBuffers() *blockBuffers {
	return &blockBuffers{
		s:   make([]float32, qBlock*kBlock),
		mx:  make([]float32, qBlock),
		sum: make([]float32, qBlock),
	}
}

// attendBlocks écrit dans o (n×D) l'attention de q sur k, v (n×D chacun).
// window < 0 : attention globale ; sinon, |i − j| ≤ window.
func attendBlocks(o, q, k, v []float32, n, D, window int, scale float32, b *blockBuffers) {
	negInf := float32(math.Inf(-1))
	for i0 := 0; i0 < n; i0 += qBlock {
		i1 := min(n, i0+qBlock)
		rows := i1 - i0
		ob := o[i0*D : i1*D]
		clear(ob)
		mx, sum := b.mx[:rows], b.sum[:rows]
		for r := range mx {
			mx[r], sum[r] = negInf, 0
		}
		j0, j1 := 0, n
		if window >= 0 {
			j0, j1 = max(0, i0-window), min(n, i1-1+window+1)
		}
		for jb := j0; jb < j1; jb += kBlock {
			je := min(j1, jb+kBlock)
			cols := je - jb
			sc := b.s[:rows*cols]
			linalg.MatMulSerial(sc, q[i0*D:i1*D], k[jb*D:je*D], rows, D, cols, false, true, false)
			for r := 0; r < rows; r++ {
				i := i0 + r
				row := sc[r*cols : (r+1)*cols]
				// Colonnes de la fenêtre dans ce bloc : [cs, ce).
				cs, ce := 0, cols
				if window >= 0 {
					cs, ce = max(0, i-window-jb), min(cols, i+window+1-jb)
				}
				if cs >= ce {
					clear(row) // aucune clé de ce bloc dans la fenêtre
					continue
				}
				valid := row[cs:ce]
				linalg.Scale(valid, scale)
				bm := linalg.MaxOf(valid)
				m := max(mx[r], bm)
				if corr := exp32(mx[r] - m); corr != 1 {
					sum[r] *= corr
					linalg.Scale(ob[r*D:(r+1)*D], corr)
				}
				mx[r] = m
				sum[r] += linalg.ExpShift(valid, m)
				clear(row[:cs])
				clear(row[ce:])
			}
			// o += P·V du bloc.
			linalg.MatMulSerial(ob, sc, v[jb*D:je*D], rows, cols, D, false, false, true)
		}
		for r := 0; r < rows; r++ {
			linalg.Scale(ob[r*D:(r+1)*D], 1/sum[r])
		}
	}
}

// RemapVocabulary réduit la table d'embeddings à un nouveau vocabulaire :
// newID[ancien] est le nouvel id d'un token gardé, -1 pour un token retiré.
// Les ids gardés doivent former [0, size).
func (m *Model) RemapVocabulary(newID []int32, size int) error {
	if len(newID) != m.Cfg.Vocab {
		return fmt.Errorf("modernbert: correspondance de %d ids pour un vocabulaire de %d", len(newID), m.Cfg.Vocab)
	}
	if pad := newID[m.Cfg.PadID]; pad < 0 {
		return fmt.Errorf("modernbert: le token de padding est retiré")
	}
	H := m.Cfg.Hidden
	w := make([]float32, size*H)
	for old, id := range newID {
		if id >= 0 {
			if int(id) >= size {
				return fmt.Errorf("modernbert: id %d hors du nouveau vocabulaire", id)
			}
			m.embRow(int32(old), w[int(id)*H:(int(id)+1)*H])
		}
	}
	m.Cfg.PadID = newID[m.Cfg.PadID]
	m.Cfg.Vocab = size
	m.Emb.W, m.Emb.Shape, m.embTable = w, []int{size, H}, nil
	return nil
}
