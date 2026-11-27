package indecis

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"time"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/internal/modernbert"
	"github.com/bornholm/indecis/optim"
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
//
// With opts.Teacher, an embedding model of the same width and tokenizer,
// the model also learns the teacher's embeddings: the contrastive loss
// weighs 1 − Distill; Distill weighs the divergence KL(teacher ‖ model)
// between the distributions of similarities of each text over the batch's
// options, softened by DistillTemp and multiplied by DistillTemp²; and
// DistillHidden weighs the alignment 1 − cos(model(x), teacher(x)),
// averaged over the texts and the options of the batch. The teacher
// encodes the texts in inference (int8 if it was loaded so) and keeps
// their embeddings in a cache of 128 MB at most (teacherCacheBytes): a
// text is encoded again only once the cache has dropped it.
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
	var tc *embedTeacher
	if opts.Teacher == nil && (opts.Distill != 0 || opts.DistillHidden != 0) {
		return fmt.Errorf("indecis: Distill and DistillHidden require a Teacher")
	}
	if opts.Teacher != nil {
		if err := m.checkTeacherEncoder(opts.Teacher); err != nil {
			return err
		}
		if opts.Distill < 0 || opts.Distill > 1 || opts.DistillHidden < 0 || opts.DistillTemp < 0 {
			return fmt.Errorf("indecis: Distill must lie in [0, 1], DistillHidden and DistillTemp be non-negative (DistillTemp 0: 1)")
		}
		if opts.Distill > 0 || opts.DistillHidden > 0 {
			tc = newEmbedTeacher(opts.Teacher, opts.Distill, opts.DistillTemp, opts.DistillHidden)
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
			loss, n, err := m.choiceStep(batches[bi], scale, opts.Symmetric, tc, grads)
			if err != nil {
				return err
			}
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

// embedTeacher is the teacher of FitEmbeddings and its weights (see
// FitEmbeddings), with the embeddings it computed most recently.
type embedTeacher struct {
	t               *Model
	kl, temp, align float64
	units           *lru // normalized embedding of each text
}

// teacherCacheBytes bounds the teacher's embeddings kept between steps:
// enough to read a corpus of tens of thousands of texts once per training,
// not the whole of a larger one.
const teacherCacheBytes = 128 << 20

func newEmbedTeacher(t *Model, kl, temp, align float64) *embedTeacher {
	if temp <= 0 {
		temp = 1
	}
	n := max(1, teacherCacheBytes/(4*t.enc.Cfg.Hidden))
	return &embedTeacher{t: t, kl: kl, temp: temp, align: align, units: newLRU(n)}
}

// embed returns the teacher's normalized embeddings of the texts, laid end
// to end, from the token ids the model reads.
func (tc *embedTeacher) embed(texts []string, seqs [][]int32) ([]float32, error) {
	H := tc.t.enc.Cfg.Hidden
	out := make([]float32, len(texts)*H)
	var todo []int
	for i, t := range texts {
		if u, ok := tc.units.get(textKey(t)); ok {
			copy(out[i*H:(i+1)*H], u)
		} else {
			todo = append(todo, i)
		}
	}
	if len(todo) > 0 {
		batch := make([][]int32, len(todo))
		for k, i := range todo {
			batch[k] = seqs[i]
		}
		pooled, err := tc.t.enc.EncodeSeqs(batch)
		if err != nil {
			return nil, fmt.Errorf("indecis: teacher: %w", err)
		}
		for k, i := range todo {
			u := normalized(pooled[k*H : (k+1)*H])
			copy(out[i*H:(i+1)*H], u)
			tc.units.put(textKey(texts[i]), u)
		}
	}
	return out, nil
}

// textKey identifies a text in the teacher's cache by 8 bytes (64-bit
// FNV-1a) rather than by the text itself: teacherCacheBytes then bounds
// what the cache holds, whatever the length of the documents. With the
// cache full (87,000 entries at width 384), a text read finds another's
// key with a probability of about 5e-15, and the collision would only
// change the teacher's embedding of that text.
func textKey(t string) string {
	h := fnv.New64a()
	h.Write([]byte(t))
	return string(h.Sum(nil))
}

// normalized returns v / |v|.
func normalized(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(max(n, 1e-24)))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// choiceStep computes the loss of a batch and accumulates the gradients of
// the two passes (texts, options); tc, if not nil, adds the distillation
// terms.
func (m *Model) choiceStep(b ChoiceBatch, scale float64, symmetric bool, tc *embedTeacher, grads *modernbert.Grads) (float64, int, error) {
	H := m.enc.Cfg.Hidden
	var teacherUnits [2][]float32 // teacher embeddings of the texts, of the options
	var teacherErr error
	encode := func(side int, texts []string) (modernbert.Batch, *modernbert.State, []float32, []float32) {
		seqs := make([][]int32, len(texts))
		for i, t := range texts {
			seqs[i] = m.tok.EncodeMax(t, m.maxLen)
		}
		if tc != nil && teacherErr == nil {
			teacherUnits[side], teacherErr = tc.embed(texts, seqs)
		}
		batch := modernbert.NewBatch(seqs, m.enc.Cfg.PadID)
		s, err := m.enc.Forward(batch)
		if err != nil {
			panic(err) // the ids come from the model's tokenizer
		}
		pooled := m.enc.Pool(s.Hidden, batch)
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
	tb, ts, a, an := encode(0, b.Texts)
	cb, cs, c, cn := encode(1, ctxs)
	if teacherErr != nil {
		return 0, 0, teacherErr
	}

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
	if tc != nil {
		weight *= 1 - tc.kl
		if tc.kl > 0 {
			loss += tc.klLossGrad(S, dS, masked, teacherUnits, nt, nc, H)
		}
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
	if tc != nil && tc.align > 0 {
		// Alignment: align * mean of 1 − u·t over the texts and options.
		w := tc.align / float64(nt+nc)
		for side, u := range [2][]float32{a, c} {
			d := [2][]float32{da, dc}[side]
			t := teacherUnits[side]
			for i := 0; i < len(u)/H; i++ {
				var dot float64
				for k := 0; k < H; k++ {
					dot += float64(u[i*H+k]) * float64(t[i*H+k])
					d[i*H+k] -= float32(w) * t[i*H+k]
				}
				loss += w * (1 - dot)
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
	m.enc.Backward(ts, m.enc.PoolBackward(unnorm(da, a, an, nt), tb), grads)
	m.enc.Backward(cs, m.enc.PoolBackward(unnorm(dc, c, cn, nc), cb), grads)
	tokens := 0
	for _, n := range tb.Lens {
		tokens += n
	}
	for _, n := range cb.Lens {
		tokens += n
	}
	return loss, tokens, nil
}

// klLossGrad adds to dS (gradient with respect to the model's scaled
// similarities S) that of kl * T² * KL(p_teacher ‖ p_model), averaged over
// the texts, the distributions being softmax(S/T) over the options that are
// not masked; it returns the loss.
func (tc *embedTeacher) klLossGrad(S, dS []float64, masked []bool, units [2][]float32, nt, nc, H int) float64 {
	T := tc.temp
	ts := tc.t.embedScale()
	var loss float64
	st, ss := make([]float64, nc), make([]float64, nc)
	for i := 0; i < nt; i++ {
		mt, ms := math.Inf(-1), math.Inf(-1)
		for j := 0; j < nc; j++ {
			if masked[i*nc+j] {
				continue
			}
			var dot float64
			for k := 0; k < H; k++ {
				dot += float64(units[0][i*H+k]) * float64(units[1][j*H+k])
			}
			st[j], ss[j] = ts*dot/T, S[i*nc+j]/T
			mt, ms = max(mt, st[j]), max(ms, ss[j])
		}
		var zt, zs float64
		for j := 0; j < nc; j++ {
			if !masked[i*nc+j] {
				zt += math.Exp(st[j] - mt)
				zs += math.Exp(ss[j] - ms)
			}
		}
		for j := 0; j < nc; j++ {
			if masked[i*nc+j] {
				continue
			}
			lpt, lps := st[j]-mt-math.Log(zt), ss[j]-ms-math.Log(zs)
			pt := math.Exp(lpt)
			loss += tc.kl * T * T * pt * (lpt - lps) / float64(nt)
			dS[i*nc+j] += tc.kl * T * (math.Exp(lps) - pt) / float64(nt)
		}
	}
	return loss
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
