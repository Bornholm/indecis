package indecis

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/bornholm/indecis/internal/modernbert"
	"github.com/bornholm/indecis/internal/optim"
)

// Choix par plongements, pour des options données à l'inférence.
//
// ChooseNearest compare le plongement du texte à celui de chaque option.
// C'est rapide (une passe par texte, les options se calculent une fois) et
// le backbone sait déjà le faire sans entraînement. FitEmbeddings l'affine
// avec l'objectif même de l'inférence : pour chaque texte d'un lot, préférer
// la bonne option parmi toutes celles du lot (entropie croisée sur les
// cosinus mis à l'échelle), comme la perte MultipleNegativesRanking de
// sentence-transformers.

// ChoiceBatch est un lot d'entraînement : des textes et une liste d'options
// communes. Correct[i] donne les options justes du texte i ; la première est
// la cible, les autres sont écartées de la comparaison (une option juste
// n'est pas une erreur).
type ChoiceBatch struct {
	Texts      []string
	Candidates []Candidate
	Correct    [][]int
}

// DefaultEmbedScale multiplie les cosinus avant le softmax (inverse d'une
// température), la valeur usuelle de sentence-transformers.
const DefaultEmbedScale = 20

// FitEmbeddings affine l'encodeur pour ChooseNearest. opts.HeadLR et
// opts.Dropout sont ignorés : aucune tête n'intervient. opts.Symmetric
// ajoute le sens option → textes à la perte.
func (m *Model) FitEmbeddings(ctx context.Context, batches []ChoiceBatch, opts TrainOptions) error {
	if opts.Epochs <= 0 || len(batches) == 0 {
		return fmt.Errorf("indecis: Epochs positif et au moins un lot requis")
	}
	for i, b := range batches {
		if len(b.Correct) != len(b.Texts) || len(b.Candidates) < 2 {
			return fmt.Errorf("indecis: lot %d mal formé", i)
		}
		for _, c := range b.Correct {
			if len(c) == 0 {
				return fmt.Errorf("indecis: lot %d : texte sans option juste", i)
			}
			for _, j := range c {
				if j < 0 || j >= len(b.Candidates) {
					return fmt.Errorf("indecis: lot %d : option %d hors liste", i, j)
				}
			}
		}
	}
	scale := m.embedScale()
	H := m.enc.Cfg.Hidden
	m.enc.Materialize()
	defer m.enc.Invalidate()
	m.embedCache.clear() // les plongements vont changer
	grads := m.enc.EnableGrad()
	var dense []optim.Dense
	for _, p := range m.enc.Params() {
		if p != m.enc.Emb {
			dense = append(dense, optim.Dense{W: p.W, G: p.G, Decay: len(p.Shape) == 2})
		}
	}
	sparse := []optim.Sparse{{W: m.enc.Emb.W, Width: H, Rows: grads.Emb.Rows}}
	opt := optim.New(optim.Config{LR: opts.LR, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: opts.WeightDecay})

	total := len(batches) * opts.Epochs
	warmup := int(math.Round(opts.Warmup * float64(total)))
	rng := rand.New(rand.NewSource(opts.Seed))
	order := make([]int, len(batches))
	for i := range order {
		order[i] = i
	}
	start := time.Now()
	step, tokens := 0, 0
	for epoch := 0; epoch < opts.Epochs; epoch++ {
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		for _, bi := range order {
			if err := ctx.Err(); err != nil {
				return err
			}
			loss, n := m.choiceStep(batches[bi], scale, opts.Symmetric, grads)
			tokens += n
			if opts.ClipNorm > 0 {
				optim.ClipGradNorm(opts.ClipNorm, dense, sparse)
			}
			f := max(0, float64(total-step)/float64(max(1, total-warmup)))
			if step < warmup {
				f = float64(step+1) / float64(warmup+1)
			}
			opt.Step(opts.LR*f, dense, sparse)
			m.enc.ZeroGrad(grads)
			m.enc.Invalidate()
			step++
			if opts.Progress != nil {
				opts.Progress(Progress{Epoch: epoch, Step: step, Steps: total, Loss: loss, LR: opts.LR * f, Tokens: tokens, Elapsed: time.Since(start)})
			}
		}
	}
	m.info.Steps += step
	return nil
}

// choiceStep calcule la perte d'un lot et accumule les gradients des deux
// passes (textes, options).
func (m *Model) choiceStep(b ChoiceBatch, scale float64, symmetric bool, grads *modernbert.Grads) (float64, int) {
	H := m.enc.Cfg.Hidden
	encode := func(texts []string) (modernbert.Batch, *modernbert.State, []float32, []float32) {
		seqs := make([][]int32, len(texts))
		for i, t := range texts {
			seqs[i] = m.tok.EncodeMax(t, m.maxLen)
		}
		batch := modernbert.NewBatch(seqs, m.enc.Cfg.PadID)
		s, err := m.enc.Forward(batch)
		if err != nil {
			panic(err) // les ids viennent du tokenizer du modèle
		}
		pooled := modernbert.MeanPool(s.Hidden, batch, H)
		norms := make([]float32, len(texts))
		unit := make([]float32, len(pooled))
		for i := range texts {
			var n float64
			for _, v := range pooled[i*H : (i+1)*H] {
				n += float64(v) * float64(v)
			}
			norms[i] = float32(math.Sqrt(max(n, 1e-24)))
			for k := 0; k < H; k++ {
				unit[i*H+k] = pooled[i*H+k] / norms[i]
			}
		}
		return batch, s, unit, norms
	}
	ctxs := make([]string, len(b.Candidates))
	for i, c := range b.Candidates {
		ctxs[i] = CandidateContext(c)
	}
	tb, ts, a, an := encode(b.Texts)
	cb, cs, c, cn := encode(ctxs)

	nt, nc := len(b.Texts), len(ctxs)
	// Similarités mises à l'échelle, et paires écartées : une option juste
	// autre que la cible n'est ni un positif ni un négatif.
	S := make([]float64, nt*nc)
	masked := make([]bool, nt*nc)
	target := make([]int, nt)
	for i := 0; i < nt; i++ {
		target[i] = b.Correct[i][0]
		for _, j := range b.Correct[i][1:] {
			masked[i*nc+j] = true
		}
		for j := 0; j < nc; j++ {
			var dot float64
			for k := 0; k < H; k++ {
				dot += float64(a[i*H+k]) * float64(c[j*H+k])
			}
			S[i*nc+j] = scale * dot
		}
	}
	dS := make([]float64, nt*nc) // ∂perte/∂S
	var loss float64

	// Texte → options : entropie croisée sur les options de chaque texte.
	weight := 1.0
	if symmetric {
		weight = 0.5
	}
	for i := 0; i < nt; i++ {
		mx := math.Inf(-1)
		for j := 0; j < nc; j++ {
			if !masked[i*nc+j] {
				mx = max(mx, S[i*nc+j])
			}
		}
		var sum float64
		for j := 0; j < nc; j++ {
			if !masked[i*nc+j] {
				sum += math.Exp(S[i*nc+j] - mx)
			}
		}
		loss += weight * -(S[i*nc+target[i]] - mx - math.Log(sum)) / float64(nt)
		for j := 0; j < nc; j++ {
			if masked[i*nc+j] {
				continue
			}
			g := math.Exp(S[i*nc+j]-mx) / sum
			if j == target[i] {
				g--
			}
			dS[i*nc+j] += weight * g / float64(nt)
		}
	}

	// Option → textes, comme CLM : chaque option cible d'au moins un texte
	// doit préférer ses textes aux autres. Plusieurs textes peuvent la viser :
	// la probabilité à maximiser est leur somme.
	if symmetric {
		var cols []int
		for j := 0; j < nc; j++ {
			for i := 0; i < nt; i++ {
				if target[i] == j {
					cols = append(cols, j)
					break
				}
			}
		}
		for _, j := range cols {
			mx := math.Inf(-1)
			for i := 0; i < nt; i++ {
				if !masked[i*nc+j] {
					mx = max(mx, S[i*nc+j])
				}
			}
			var all, pos float64
			for i := 0; i < nt; i++ {
				if masked[i*nc+j] {
					continue
				}
				e := math.Exp(S[i*nc+j] - mx)
				all += e
				if target[i] == j {
					pos += e
				}
			}
			loss += weight * -math.Log(pos/all) / float64(len(cols))
			for i := 0; i < nt; i++ {
				if masked[i*nc+j] {
					continue
				}
				e := math.Exp(S[i*nc+j] - mx)
				g := e / all
				if target[i] == j {
					g -= e / pos
				}
				dS[i*nc+j] += weight * g / float64(len(cols))
			}
		}
	}

	da := make([]float32, len(a))
	dc := make([]float32, len(c))
	for i := 0; i < nt; i++ {
		for j := 0; j < nc; j++ {
			g := float32(scale * dS[i*nc+j])
			if g == 0 {
				continue
			}
			for k := 0; k < H; k++ {
				da[i*H+k] += g * c[j*H+k]
				dc[j*H+k] += g * a[i*H+k]
			}
		}
	}
	// Retour à travers la normalisation : dx = (du − u·(u·du)) / |x|.
	unnorm := func(d, u, norms []float32, n int) []float32 {
		out := make([]float32, len(d))
		for i := 0; i < n; i++ {
			var p float32
			for k := 0; k < H; k++ {
				p += u[i*H+k] * d[i*H+k]
			}
			for k := 0; k < H; k++ {
				out[i*H+k] = (d[i*H+k] - u[i*H+k]*p) / norms[i]
			}
		}
		return out
	}
	m.enc.Backward(ts, modernbert.MeanPoolBackward(unnorm(da, a, an, nt), tb, H), grads)
	m.enc.Backward(cs, modernbert.MeanPoolBackward(unnorm(dc, c, cn, nc), cb, H), grads)
	tokens := 0
	for _, n := range tb.Lens {
		tokens += n
	}
	for _, n := range cb.Lens {
		tokens += n
	}
	return loss, tokens
}

func (m *Model) embedScale() float64 {
	if m.info.EmbedScale > 0 {
		return m.info.EmbedScale
	}
	return DefaultEmbedScale
}

// CandidateSet est une liste d'options préparée pour ChooseIn : le
// plongement de chaque option est calculé une fois. Il reste valable tant
// que le modèle n'est pas réentraîné.
type CandidateSet struct {
	candidates []Candidate
	protos     [][]float32
}

// Candidates retourne les options de la liste.
func (s *CandidateSet) Candidates() []Candidate { return s.candidates }

// PrepareCandidates calcule le prototype de chaque option : la moyenne,
// normalisée, du plongement de son nom et de sa description et de ceux de
// ses exemples.
func (m *Model) PrepareCandidates(ctx context.Context, candidates []Candidate) (*CandidateSet, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("indecis: aucune option")
	}
	seen := map[string]bool{}
	var texts []string
	owner := []int{}
	for i, c := range candidates {
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("indecis: option %q vide ou en double", c.Name)
		}
		seen[c.Name] = true
		texts = append(texts, CandidateContext(c))
		owner = append(owner, i)
		for _, e := range c.Examples {
			texts = append(texts, e)
			owner = append(owner, i)
		}
	}
	vecs, err := m.Embed(ctx, texts...)
	if err != nil {
		return nil, err
	}
	H := m.enc.Cfg.Hidden
	sums := make([][]float64, len(candidates))
	for i := range sums {
		sums[i] = make([]float64, H)
	}
	for k, v := range vecs {
		for d, x := range v {
			sums[owner[k]][d] += float64(x)
		}
	}
	set := &CandidateSet{candidates: append([]Candidate(nil), candidates...), protos: make([][]float32, len(candidates))}
	for i, s := range sums {
		var n float64
		for _, x := range s {
			n += x * x
		}
		inv := 1 / math.Sqrt(max(n, 1e-24))
		p := make([]float32, H)
		for d, x := range s {
			p[d] = float32(x * inv)
		}
		set.protos[i] = p
	}
	return set, nil
}

// ChooseNearest choisit, pour chaque texte, l'option dont le prototype est
// le plus proche (voir PrepareCandidates et ChooseIn).
func (m *Model) ChooseNearest(ctx context.Context, candidates []Candidate, texts ...string) ([]Answer, error) {
	set, err := m.PrepareCandidates(ctx, candidates)
	if err != nil {
		return nil, err
	}
	return m.ChooseIn(ctx, set, texts...)
}

// ChooseIn classe chaque texte parmi les options d'une liste préparée.
// Answer.Probs est le softmax des cosinus mis à l'échelle (Info.EmbedScale,
// DefaultEmbedScale par défaut). Answer.Score est le cosinus de l'option
// retenue : un seuil sur ce cosinus, réglé sur des exemples, sépare les
// textes qu'aucune option ne décrit.
func (m *Model) ChooseIn(ctx context.Context, set *CandidateSet, texts ...string) ([]Answer, error) {
	te, err := m.embed(ctx, texts, false)
	if err != nil {
		return nil, err
	}
	scale := m.embedScale()
	out := make([]Answer, len(texts))
	for i, t := range te {
		cos := make([]float64, len(set.protos))
		z := make([]float64, len(set.protos))
		for j, c := range set.protos {
			var dot float64
			for k := range t {
				dot += float64(t[k]) * float64(c[k])
			}
			cos[j], z[j] = dot, scale*dot
		}
		p := softmax(z)
		a := Answer{Kind: Choice, Probs: make(map[string]float64, len(p))}
		best := 0
		for j, v := range p {
			a.Probs[set.candidates[j].Name] = v
			if v > p[best] {
				best = j
			}
		}
		a.Choice, a.Confidence, a.Score = set.candidates[best].Name, p[best], cos[best]
		a.Margin = margin(p, best)
		out[i] = a
	}
	return out, nil
}
