package vision

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/optim"
)

// Head answers learned questions (an indecis.Schema) from the patch
// features of a frozen image encoder. It is spatial: each patch is
// projected to K values with shared weights and a ReLU, then the outputs
// read the K values of every patch with weights of their own, so a
// question can depend on where something is in the image ("the enemy is
// to the left of the crosshair"), which the pooled embedding blurs. They
// also read the maximum of each of the K values over the image, for
// questions about anywhere ("an enemy is in sight").
type Head struct {
	Schema indecis.Schema
	// T patches of H features, projected to K values each.
	T, H, K int
	// Layer is the encoder layer whose patch features the head reads: 0
	// for the final ones, n for the hidden states after n layers (the
	// encoder then stops there when only learned questions are asked).
	Layer  int
	W1, B1 []float32 // [K, H], [K]
	W2, B2 []float32 // [O, T·K + K], [O]: O outputs, see outputs
}

// outputs returns the offset of each question's outputs and their total:
// one logit for a noul, one per option or level otherwise.
func (h *Head) outputs() ([]int, int) {
	offs := make([]int, len(h.Schema))
	o := 0
	for i, q := range h.Schema {
		offs[i] = o
		if q.Kind == indecis.Noul {
			o++
		} else {
			o += len(q.Options)
		}
	}
	return offs, o
}

// NewHead initializes a head for patches of shape [t, hidden].
func NewHead(schema indecis.Schema, t, hidden, k int, seed int64) (*Head, error) {
	if err := schema.Validate(); err != nil {
		return nil, err
	}
	h := &Head{Schema: schema, T: t, H: hidden, K: k}
	_, o := h.outputs()
	r := rand.New(rand.NewSource(seed))
	randn := func(n int, std float64) []float32 {
		w := make([]float32, n)
		for i := range w {
			w[i] = float32(r.NormFloat64() * std)
		}
		return w
	}
	h.W1, h.B1 = randn(k*hidden, math.Sqrt(2/float64(hidden))), make([]float32, k)
	h.W2, h.B2 = randn(o*h.features(), math.Sqrt(1/float64(h.features()))), make([]float32, o)
	return h, nil
}

// features is the width of what the outputs read: T·K spatial values and
// K maxima.
func (h *Head) features() int { return h.T*h.K + h.K }

// logits computes the outputs for n examples, x [n·T, H]: z [n·T, K]
// holds the projected patches after the ReLU, f [n, T·K + K] the features
// the outputs read (z, then the maximum of each value over the patches),
// arg [n·K] the patch of each maximum, l [n, O] the outputs.
func (h *Head) logits(x []float32, n int, z, f []float32, arg []int, l []float32) {
	T, H, K := h.T, h.H, h.K
	F := h.features()
	_, O := h.outputs()
	linalg.MatMul(z[:n*T*K], x[:n*T*H], h.W1, n*T, H, K, false, true, false)
	for r := 0; r < n*T; r++ {
		row := z[r*K : (r+1)*K]
		linalg.AddTo(row, h.B1)
		for j, v := range row {
			if v < 0 {
				row[j] = 0
			}
		}
	}
	for b := 0; b < n; b++ {
		fb := f[b*F : (b+1)*F]
		copy(fb, z[b*T*K:(b+1)*T*K])
		mx, am := fb[T*K:], arg[b*K:(b+1)*K]
		for k := 0; k < K; k++ {
			mx[k], am[k] = z[b*T*K+k], 0
			for t := 1; t < T; t++ {
				if v := z[b*T*K+t*K+k]; v > mx[k] {
					mx[k], am[k] = v, t
				}
			}
		}
	}
	linalg.MatMul(l[:n*O], f[:n*F], h.W2, n, F, O, false, true, false)
	for r := 0; r < n; r++ {
		linalg.AddTo(l[r*O:(r+1)*O], h.B2)
	}
}

// Decide answers the schema's questions from the patches of one image,
// [T, H] (Model.Patches).
func (h *Head) Decide(patches []float32) (indecis.Decision, error) {
	if len(patches) != h.T*h.H {
		return nil, fmt.Errorf("vision: %d patch features, expected %d×%d", len(patches), h.T, h.H)
	}
	offs, O := h.outputs()
	z := make([]float32, h.T*h.K)
	f := make([]float32, h.features())
	arg := make([]int, h.K)
	l := make([]float32, O)
	h.logits(patches, 1, z, f, arg, l)
	d := indecis.Decision{}
	for i, q := range h.Schema {
		lq := l[offs[i]:]
		a := indecis.Answer{Question: q.Name, Kind: q.Kind}
		if q.Kind == indecis.Noul {
			a.P = sigmoid(float64(lq[0]))
			a.Confidence = math.Max(a.P, 1-a.P)
			d[q.Name] = a
			continue
		}
		z := make([]float64, len(q.Options))
		for j := range z {
			z[j] = float64(lq[j])
		}
		p := softmax(z)
		best := 0
		a.Probs = make(map[string]float64, len(p))
		for j, o := range q.Options {
			a.Probs[o] = p[j]
			if p[j] > p[best] {
				best = j
			}
		}
		a.Choice, a.Confidence = q.Options[best], p[best]
		if q.Kind == indecis.Score {
			for j := range p {
				a.Score += float64(j) * p[j]
			}
		}
		d[q.Name] = a
	}
	return d, nil
}

// HeadExample is a training example: the patch features of an image in
// bfloat16 (half the memory of float32, far below the precision a head
// needs), and labels in the dataset format (see indecis.Question.Target).
type HeadExample struct {
	Patches []uint16
	Labels  map[string]any
}

// ToBF16 converts patch features to bfloat16, rounding to nearest even.
func ToBF16(x []float32) []uint16 {
	out := make([]uint16, len(x))
	for i, v := range x {
		b := math.Float32bits(v)
		b += 0x7fff + (b>>16)&1
		out[i] = uint16(b >> 16)
	}
	return out
}

func fromBF16(dst []float32, x []uint16) {
	for i, v := range x {
		dst[i] = math.Float32frombits(uint32(v) << 16)
	}
}

// HeadTrainOptions configures Fit.
type HeadTrainOptions struct {
	Epochs      int
	BatchSize   int
	LR          float64
	WeightDecay float64
	Seed        int64
	// Progress, if not nil, receives the mean loss of each epoch.
	Progress func(epoch int, loss float64)
}

// DefaultHeadTrainOptions trains for 10 epochs, batches of 32, rate 3e-3.
func DefaultHeadTrainOptions() HeadTrainOptions {
	return HeadTrainOptions{Epochs: 10, BatchSize: 32, LR: 3e-3, WeightDecay: 1e-4, Seed: 1}
}

// target holds the training targets of one example, per question (nil
// when the label is absent).
type target [][]float64

func (h *Head) targets(examples []HeadExample) ([]target, error) {
	out := make([]target, len(examples))
	for e, ex := range examples {
		if len(ex.Patches) != h.T*h.H {
			return nil, fmt.Errorf("vision: example %d: %d patch features, expected %d", e, len(ex.Patches), h.T*h.H)
		}
		t := make(target, len(h.Schema))
		for i, q := range h.Schema {
			v, ok, err := q.Target(ex.Labels[q.Name])
			if err != nil {
				return nil, fmt.Errorf("vision: example %d: %w", e, err)
			}
			if !ok {
				continue
			}
			if q.Kind == indecis.Score { // a level: one-hot over the levels
				oh := make([]float64, len(q.Options))
				oh[int(v[0])] = 1
				v = oh
			}
			t[i] = v
		}
		out[e] = t
	}
	return out, nil
}

// Fit trains the head: binary cross-entropy for noul questions, cross-
// entropy over the options or levels otherwise; an absent label does not
// count. The result is the same bit for bit from one run to the next.
func (h *Head) Fit(examples []HeadExample, opts HeadTrainOptions) error {
	if len(examples) == 0 {
		return fmt.Errorf("vision: no training examples")
	}
	targets, err := h.targets(examples)
	if err != nil {
		return err
	}
	T, H, K := h.T, h.H, h.K
	offs, O := h.outputs()
	B := max(1, opts.BatchSize)
	F := h.features()
	x := make([]float32, B*T*H)
	zbuf := make([]float64, O) // one question's logits, then probabilities
	z := make([]float32, B*T*K)
	f := make([]float32, B*F)
	arg := make([]int, B*K)
	l := make([]float32, B*O)
	dl := make([]float32, B*O)
	df := make([]float32, B*F)
	dz := make([]float32, B*T*K)
	gW1, gB1 := make([]float32, len(h.W1)), make([]float32, len(h.B1))
	gW2, gB2 := make([]float32, len(h.W2)), make([]float32, len(h.B2))
	params := []optim.Dense{{W: h.W1, G: gW1, Decay: true}, {W: h.B1, G: gB1}, {W: h.W2, G: gW2, Decay: true}, {W: h.B2, G: gB2}}
	cfg := optim.DefaultConfig(opts.LR)
	cfg.WeightDecay = opts.WeightDecay
	opt := optim.New(cfg)
	r := rand.New(rand.NewSource(opts.Seed))
	order := make([]int, len(examples))
	for i := range order {
		order[i] = i
	}
	for epoch := 1; epoch <= opts.Epochs; epoch++ {
		r.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		var total float64
		for start := 0; start < len(order); start += B {
			batch := order[start:min(start+B, len(order))]
			n := len(batch)
			for b, e := range batch {
				fromBF16(x[b*T*H:(b+1)*T*H], examples[e].Patches)
			}
			h.logits(x, n, z, f, arg, l)
			clear(dl[:n*O])
			for b, e := range batch {
				lb, db := l[b*O:(b+1)*O], dl[b*O:(b+1)*O]
				for i, q := range h.Schema {
					t := targets[e][i]
					if t == nil {
						continue
					}
					o := offs[i]
					if q.Kind == indecis.Noul {
						p := sigmoid(float64(lb[o]))
						total += -(t[0]*math.Log(max(p, 1e-12)) + (1-t[0])*math.Log(max(1-p, 1e-12)))
						db[o] = float32((p - t[0]) / float64(n))
						continue
					}
					p := zbuf[:len(t)]
					for j := range p {
						p[j] = float64(lb[o+j])
					}
					softmaxInto(p, p)
					for j := range p {
						if t[j] > 0 {
							total -= t[j] * math.Log(max(p[j], 1e-12))
						}
						db[o+j] = float32((p[j] - t[j]) / float64(n))
					}
				}
			}
			// Backward: the encoder is frozen, only the head learns.
			linalg.MatMul(gW2, dl[:n*O], f[:n*F], O, n, F, true, false, false)
			clear(gB2)
			for b := 0; b < n; b++ {
				linalg.AddTo(gB2, dl[b*O:(b+1)*O])
			}
			linalg.MatMul(df[:n*F], dl[:n*O], h.W2, n, O, F, false, false, false)
			// The spatial part flows back as is, each maximum to its patch.
			for b := 0; b < n; b++ {
				copy(dz[b*T*K:(b+1)*T*K], df[b*F:b*F+T*K])
				for k := 0; k < K; k++ {
					dz[b*T*K+arg[b*K+k]*K+k] += df[b*F+T*K+k]
				}
			}
			for i := range dz[:n*T*K] {
				if z[i] <= 0 {
					dz[i] = 0
				}
			}
			linalg.MatMul(gW1, dz[:n*T*K], x[:n*T*H], K, n*T, H, true, false, false)
			clear(gB1)
			for row := 0; row < n*T; row++ {
				linalg.AddTo(gB1, dz[row*K:(row+1)*K])
			}
			opt.Step(opts.LR, params, nil)
		}
		if opts.Progress != nil {
			opts.Progress(epoch, total/float64(len(examples)))
		}
	}
	return nil
}

// Evaluate returns, per question, the share of labeled examples answered
// right: p >= 0.5 against the label for a noul, the most likely option or
// level against the labeled one otherwise.
func (h *Head) Evaluate(examples []HeadExample) (map[string]float64, error) {
	targets, err := h.targets(examples)
	if err != nil {
		return nil, err
	}
	right := make([]int, len(h.Schema))
	count := make([]int, len(h.Schema))
	x := make([]float32, h.T*h.H)
	for e, ex := range examples {
		fromBF16(x, ex.Patches)
		d, err := h.Decide(x)
		if err != nil {
			return nil, err
		}
		for i, q := range h.Schema {
			t := targets[e][i]
			if t == nil {
				continue
			}
			count[i]++
			a := d[q.Name]
			if q.Kind == indecis.Noul {
				if (a.P >= 0.5) == (t[0] >= 0.5) {
					right[i]++
				}
				continue
			}
			best := 0
			for j := range t {
				if t[j] > t[best] {
					best = j
				}
			}
			if a.Choice == q.Options[best] {
				right[i]++
			}
		}
	}
	out := map[string]float64{}
	for i, q := range h.Schema {
		if count[i] > 0 {
			out[q.Name] = float64(right[i]) / float64(count[i])
		}
	}
	return out, nil
}
