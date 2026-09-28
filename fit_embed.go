package indecis

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/internal/modernbert"
	"github.com/bornholm/indecis/internal/optim"
)

// Choice by embeddings, for options given at inference time.
//
// ChooseNearest compares the embedding of the text to that of each option.
// It is fast (one pass per text, the options are computed once) and the
// backbone already knows how to do it without training. FitEmbeddings
// refines it with the very objective of inference: for each text of a
// batch, prefer the right option among all those in the batch (cross
// entropy on the scaled cosines), like the MultipleNegativesRanking loss
// of sentence-transformers.

// ChoiceBatch is a training batch: texts and a list of shared options.
// Correct[i] gives the right options for text i; the first is the target,
// the others are excluded from the comparison (a correct option is not an
// error).
type ChoiceBatch struct {
	Texts      []string
	Candidates []Candidate
	Correct    [][]int
}

// DefaultEmbedScale multiplies the cosines before the softmax (inverse of a
// temperature), the usual value in sentence-transformers.
const DefaultEmbedScale = 20

// FitEmbeddings fine-tunes the encoder for ChooseNearest. opts.HeadLR and
// opts.Dropout are ignored: no head is involved. opts.Symmetric adds the
// option -> texts direction to the loss.
func (m *Model) FitEmbeddings(ctx context.Context, batches []ChoiceBatch, opts TrainOptions) error {
	if opts.Epochs <= 0 || len(batches) == 0 {
		return fmt.Errorf("indecis: Epochs must be positive and at least one batch is required")
	}
	for i, b := range batches {
		if len(b.Correct) != len(b.Texts) || len(b.Candidates) < 2 {
			return fmt.Errorf("indecis: malformed batch %d", i)
		}
		for _, c := range b.Correct {
			if len(c) == 0 {
				return fmt.Errorf("indecis: batch %d: text without a correct option", i)
			}
			for _, j := range c {
				if j < 0 || j >= len(b.Candidates) {
					return fmt.Errorf("indecis: batch %d: option %d out of list", i, j)
				}
			}
		}
	}
	scale := m.embedScale()
	H := m.enc.Cfg.Hidden
	m.enc.Materialize()
	defer m.enc.Invalidate()
	m.embedCache.clear() // the embeddings are about to change
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

// choiceStep computes the loss of a batch and accumulates the gradients of
// the two passes (texts, options).
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
			panic(err) // the ids come from the model's tokenizer
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
	// Scaled similarities, and excluded pairs: a correct option other than
	// the target is neither a positive nor a negative.
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
	dS := make([]float64, nt*nc) // dLoss/dS
	var loss float64

	// Text -> options: cross entropy on the options of each text.
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

	// Option -> texts, like CLM: each option that is the target of at least
	// one text must prefer its texts over the others. Several texts can
	// target it: the probability to maximize is their sum.
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
	// Back through normalization: dx = (du - u*(u.du)) / |x|.
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

// CandidateSet is a list of options prepared for ChooseIn: the embedding of
// each option is computed once. It remains valid as long as the model is
// not retrained.
type CandidateSet struct {
	candidates []Candidate
	protos     [][]float32
}

// Candidates returns the options of the list.
func (s *CandidateSet) Candidates() []Candidate { return s.candidates }

// PrepareCandidates computes the prototype of each option: the normalized
// average of the embedding of its name and its description and those of
// its examples.
func (m *Model) PrepareCandidates(ctx context.Context, candidates []Candidate) (*CandidateSet, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("indecis: no options")
	}
	seen := map[string]bool{}
	var texts []string
	owner := []int{}
	for i, c := range candidates {
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("indecis: option %q empty or duplicate", c.Name)
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

// ChooseNearest picks, for each text, the option whose prototype is
// closest (see PrepareCandidates and ChooseIn).
func (m *Model) ChooseNearest(ctx context.Context, candidates []Candidate, texts ...string) ([]Answer, error) {
	set, err := m.PrepareCandidates(ctx, candidates)
	if err != nil {
		return nil, err
	}
	return m.ChooseIn(ctx, set, texts...)
}

// ChooseIn ranks each text among the options of a prepared list.
// Answer.Probs is the softmax of the scaled cosines (Info.EmbedScale,
// DefaultEmbedScale by default). Answer.Score is the cosine of the chosen
// option: a threshold on this cosine, tuned on examples, separates texts
// that no option describes.
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

// ChoiceBatches forms the FitEmbeddings batches from labeled examples: for
// each question, the examples that carry its label are grouped into
// batches of size texts, with all the question's options as the batch's
// options. A label is an option name, a distribution (the most probable
// option is kept), a level index (Score), or a boolean (Noul: the first
// option for true, the second for false).
func ChoiceBatches(examples []dataset.Example, questions []OpenQuestion, size int, seed int64) []ChoiceBatch {
	rng := rand.New(rand.NewSource(seed))
	var out []ChoiceBatch
	for _, q := range questions {
		type item struct {
			text string
			pos  int
		}
		var items []item
		for _, e := range examples {
			if v, ok := e.Labels[q.Name]; ok {
				if j := q.OptionIndex(v); j >= 0 {
					items = append(items, item{e.Text, j})
				}
			}
		}
		rng.Shuffle(len(items), func(i, j int) { items[i], items[j] = items[j], items[i] })
		for s := 0; s < len(items); s += size {
			b := ChoiceBatch{Candidates: q.Options}
			for _, it := range items[s:min(len(items), s+size)] {
				b.Texts = append(b.Texts, it.text)
				b.Correct = append(b.Correct, []int{it.pos})
			}
			out = append(out, b)
		}
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// OptionIndex returns the index of the option that a label designates (see
// ChoiceBatches), or -1.
func (q OpenQuestion) OptionIndex(v any) int {
	switch x := v.(type) {
	case bool:
		if x {
			return 0
		}
		return 1
	case string:
		for i, o := range q.Options {
			if o.Name == x {
				return i
			}
		}
	case float64:
		if q.Kind == Score && x == math.Trunc(x) && int(x) >= 0 && int(x) < len(q.Options) {
			return int(x)
		}
		if q.Kind == Noul {
			if x >= 0.5 {
				return 0
			}
			return 1
		}
	case map[string]any:
		best, bp := -1, -1.0
		for i, o := range q.Options {
			if p, ok := x[o.Name].(float64); ok && p > bp {
				best, bp = i, p
			}
		}
		return best
	}
	return -1
}
