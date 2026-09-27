package indecis

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/internal/modernbert"
	"github.com/bornholm/indecis/internal/optim"
)

// TrainOptions règle le fine-tuning.
type TrainOptions struct {
	Epochs    int
	BatchSize int
	// LR est le taux d'apprentissage de l'encodeur, HeadLR celui des têtes,
	// plus élevé : elles partent de zéro.
	LR, HeadLR  float64
	WeightDecay float64
	// Warmup est la fraction des pas passée à monter le taux linéairement,
	// qui décroît ensuite linéairement jusqu'à zéro.
	Warmup float64
	// ClipNorm borne la norme globale des gradients (0 : pas d'écrêtage).
	ClipNorm float64
	// Dropout s'applique au vecteur poolé, avant les têtes.
	Dropout float64
	Seed    int64
	// Symmetric (FitEmbeddings) optimise aussi le sens option → textes :
	// chaque option d'un lot doit préférer ses textes aux autres, comme la
	// perte InfoNCE bidirectionnelle de CLM.
	Symmetric bool
	// Progress, s'il est fourni, est appelé après chaque pas.
	Progress func(Progress)
}

// DefaultTrainOptions sont des réglages de départ pour un petit encodeur.
func DefaultTrainOptions() TrainOptions {
	return TrainOptions{
		Epochs: 3, BatchSize: 16, LR: 1e-4, HeadLR: 1e-3, WeightDecay: 0.01,
		Warmup: 0.1, ClipNorm: 1, Dropout: 0.1, Seed: 1,
	}
}

// Progress décrit l'avancement de l'entraînement.
type Progress struct {
	Epoch, Step, Steps int
	Loss               float64 // perte moyenne du pas
	LR                 float64
	Tokens             int // tokens traités depuis le début
	Elapsed            time.Duration
}

// encoded est un exemple tokenisé et ses cibles, une par question (nil si
// l'étiquette manque).
type encoded struct {
	ids     []int32
	targets [][]float64
}

func (m *Model) encode(examples []dataset.Example) ([]encoded, error) {
	out := make([]encoded, len(examples))
	for i, e := range examples {
		ids, err := m.tokenize(e.Context, e.Text)
		if err != nil {
			return nil, fmt.Errorf("indecis: exemple %d : %w", i, err)
		}
		out[i].ids = ids
		out[i].targets = make([][]float64, len(m.schema))
		for name := range e.Labels {
			if m.schema.Index(name) < 0 {
				return nil, fmt.Errorf("indecis: exemple %d : question %q absente du schéma", i, name)
			}
		}
		for qi, q := range m.schema {
			t, ok, err := q.target(e.Labels[q.Name])
			if err != nil {
				return nil, fmt.Errorf("indecis: exemple %d : %w", i, err)
			}
			if ok {
				out[i].targets[qi] = t
			}
		}
	}
	return out, nil
}

// Fit fine-tune tout le modèle, encodeur compris, sur les exemples.
func (m *Model) Fit(ctx context.Context, examples []dataset.Example, opts TrainOptions) error {
	if opts.BatchSize <= 0 || opts.Epochs <= 0 {
		return fmt.Errorf("indecis: BatchSize et Epochs doivent être positifs")
	}
	data, err := m.encode(examples)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return fmt.Errorf("indecis: aucun exemple")
	}
	m.recordPriors(data)

	H := m.enc.Cfg.Hidden
	// L'entraînement modifie les poids : la table d'embeddings doit être en
	// float32, et les poids empaquetés pour l'inférence sont refaits après
	// chaque pas (Progress peut évaluer le modèle en cours de route).
	m.enc.Materialize()
	defer m.enc.Invalidate()
	m.embedCache.clear() // les plongements vont changer
	grads := m.enc.EnableGrad()
	var encDense []optim.Dense
	for _, p := range m.enc.Params() {
		if p == m.enc.Emb {
			continue
		}
		encDense = append(encDense, optim.Dense{W: p.W, G: p.G, Decay: len(p.Shape) == 2})
	}
	encSparse := []optim.Sparse{{W: m.enc.Emb.W, Width: H, Rows: grads.Emb.Rows}}
	var headDense []optim.Dense
	for _, h := range m.heads {
		h.enableGrad()
		headDense = append(headDense, optim.Dense{W: h.w, G: h.gw, Decay: true}, optim.Dense{W: h.b, G: h.gb})
	}
	encOpt := optim.New(optim.Config{LR: opts.LR, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: opts.WeightDecay})
	headOpt := optim.New(optim.Config{LR: opts.HeadLR, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: opts.WeightDecay})

	rng := rand.New(rand.NewSource(opts.Seed))
	stepsPerEpoch := (len(data) + opts.BatchSize - 1) / opts.BatchSize
	total := stepsPerEpoch * opts.Epochs
	warmup := int(math.Round(opts.Warmup * float64(total)))
	schedule := func(step int) float64 {
		if step < warmup {
			return float64(step+1) / float64(warmup+1)
		}
		return max(0, float64(total-step)/float64(max(1, total-warmup)))
	}

	start := time.Now()
	step, tokens := 0, 0
	for epoch := 0; epoch < opts.Epochs; epoch++ {
		for _, batch := range batches(data, opts.BatchSize, rng) {
			if err := ctx.Err(); err != nil {
				return err
			}
			loss, n := m.trainStep(batch, grads, opts.Dropout, rng)
			tokens += n
			all := append(append([]optim.Dense(nil), encDense...), headDense...)
			if opts.ClipNorm > 0 {
				optim.ClipGradNorm(opts.ClipNorm, all, encSparse)
			}
			f := schedule(step)
			encOpt.Step(opts.LR*f, encDense, encSparse)
			headOpt.Step(opts.HeadLR*f, headDense, nil)
			m.enc.Invalidate()
			m.enc.ZeroGrad(grads)
			for _, h := range m.heads {
				h.zeroGrad()
			}
			step++
			if opts.Progress != nil {
				opts.Progress(Progress{Epoch: epoch, Step: step, Steps: total, Loss: loss, LR: opts.LR * f, Tokens: tokens, Elapsed: time.Since(start)})
			}
		}
	}
	m.info.Steps += step
	return nil
}

// batches mélange les exemples puis regroupe des longueurs voisines : on
// trie des fenêtres de 50 lots par longueur avant de les découper. Le
// padding diminue sans que les lots deviennent prévisibles.
func batches(data []encoded, size int, rng *rand.Rand) [][]*encoded {
	idx := rng.Perm(len(data))
	window := size * 50
	var out [][]*encoded
	for w := 0; w < len(idx); w += window {
		chunk := idx[w:min(w+window, len(idx))]
		sort.SliceStable(chunk, func(a, b int) bool { return len(data[chunk[a]].ids) < len(data[chunk[b]].ids) })
		for s := 0; s < len(chunk); s += size {
			var b []*encoded
			for _, i := range chunk[s:min(s+size, len(chunk))] {
				b = append(b, &data[i])
			}
			out = append(out, b)
		}
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// trainStep calcule la perte d'un lot et accumule tous les gradients. La
// perte de chaque question est moyennée sur les exemples qui l'étiquettent,
// puis les questions sont sommées.
func (m *Model) trainStep(batch []*encoded, grads *modernbert.Grads, dropout float64, rng *rand.Rand) (float64, int) {
	H := m.enc.Cfg.Hidden
	seqs := make([][]int32, len(batch))
	tokens := 0
	for i, e := range batch {
		seqs[i] = e.ids
		tokens += len(e.ids)
	}
	b := modernbert.NewBatch(seqs, m.enc.Cfg.PadID)
	s, err := m.enc.Forward(b)
	if err != nil {
		panic(err) // les ids viennent du tokenizer du modèle
	}
	pooled := modernbert.MeanPool(s.Hidden, b, H)

	// Dropout inversé sur le vecteur poolé.
	mask := make([]float32, len(pooled))
	keep := float32(1 / (1 - dropout))
	for i := range mask {
		if dropout <= 0 || rng.Float64() >= dropout {
			mask[i] = 1
			if dropout > 0 {
				mask[i] = keep
			}
		}
	}
	x := make([]float32, len(pooled))
	for i := range x {
		x[i] = pooled[i] * mask[i]
	}

	counts := make([]int, len(m.heads))
	for _, e := range batch {
		for qi, t := range e.targets {
			if t != nil {
				counts[qi]++
			}
		}
	}
	dx := make([]float32, len(pooled))
	var loss float64
	for i, e := range batch {
		xi := x[i*H : (i+1)*H]
		for qi, h := range m.heads {
			t := e.targets[qi]
			if t == nil {
				continue
			}
			l, dz := h.lossGrad(h.logits(xi), t)
			scale := 1 / float64(counts[qi])
			loss += l * scale
			for k := range dz {
				dz[k] *= scale
			}
			h.backward(xi, dz, dx[i*H:(i+1)*H])
		}
	}
	for i := range dx {
		dx[i] *= mask[i]
	}
	m.enc.Backward(s, modernbert.MeanPoolBackward(dx, b, H), grads)
	return loss, tokens
}

func (m *Model) recordPriors(data []encoded) {
	m.info.TrainPrior = map[string]float64{}
	for qi, q := range m.schema {
		if q.Kind != Noul {
			continue
		}
		var sum float64
		n := 0
		for _, e := range data {
			if t := e.targets[qi]; t != nil {
				sum += t[0]
				n++
			}
		}
		if n > 0 {
			m.info.TrainPrior[q.Name] = sum / float64(n)
		}
	}
}

// Calibrate ajuste la température de chaque question sur des exemples
// distincts de ceux de l'entraînement, en minimisant la log-vraisemblance
// négative. Retourne les températures retenues.
//
// Une question dont les exemples de calibration sont tous parfaitement
// classés garde sa température : ces exemples ne disent rien de l'erreur du
// modèle, seulement qu'ils sont trop faciles.
func (m *Model) Calibrate(ctx context.Context, examples []dataset.Example) (map[string]float64, error) {
	data, err := m.encode(examples)
	if err != nil {
		return nil, err
	}
	logits, err := m.logits(ctx, examplesToInputs(examples), 32)
	if err != nil {
		return nil, err
	}
	for qi, h := range m.heads {
		var zs, ts [][]float64
		for i, e := range data {
			if e.targets[qi] != nil {
				zs = append(zs, logits[i][qi])
				ts = append(ts, e.targets[qi])
			}
		}
		if len(zs) == 0 {
			continue
		}
		nll := func(T float64) float64 {
			var sum float64
			scaled := make([]float64, 0, 8)
			for i, z := range zs {
				scaled = scaled[:0]
				for _, v := range z {
					scaled = append(scaled, v/T)
				}
				l, _ := h.lossGrad(scaled, ts[i])
				sum += l
			}
			return sum / float64(len(zs))
		}
		if separable(h, zs, ts) {
			// Données parfaitement séparées : la vraisemblance décroît
			// jusqu'à T → 0, et « calibrer » rendrait le modèle
			// arbitrairement sûr de lui. La température reste inchangée.
			continue
		}
		m.temps[qi] = goldenLog(nll, 0.05, 20)
	}
	return m.Temperatures(), nil
}

// separable indique si toutes les réponses sont justes et que la perte ne
// fait que baisser en rendant le modèle plus tranchant : aucun exemple ne
// tire la température vers le haut.
func separable(h *head, zs, ts [][]float64) bool {
	for i, z := range zs {
		a := h.answer(z, 1)
		t := ts[i]
		switch h.q.Kind {
		case Noul:
			if (a.P >= 0.5) != (t[0] >= 0.5) || (t[0] > 0 && t[0] < 1) {
				return false
			}
		case Choice:
			if t[indexOf(h.q.Options, a.Choice)] != maxOf(t) || maxOf(t) < 1 {
				return false
			}
		case Score:
			if a.Choice != h.q.Options[int(t[0])] {
				return false
			}
		}
	}
	return true
}

// goldenLog minimise f sur [lo, hi] par section dorée sur log(T).
func goldenLog(f func(float64) float64, lo, hi float64) float64 {
	a, b := math.Log(lo), math.Log(hi)
	const phi = 0.6180339887498949
	c, d := b-phi*(b-a), a+phi*(b-a)
	fc, fd := f(math.Exp(c)), f(math.Exp(d))
	for i := 0; i < 60; i++ {
		if fc < fd {
			b, d, fd = d, c, fc
			c = b - phi*(b-a)
			fc = f(math.Exp(c))
		} else {
			a, c, fc = c, d, fd
			d = a + phi*(b-a)
			fd = f(math.Exp(d))
		}
	}
	return math.Exp((a + b) / 2)
}
