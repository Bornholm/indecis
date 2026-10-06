package indecis

import (
	"math"
	"math/rand"
)

// head is the output layer of a question, applied to the pooled vector,
// or to each token for Spans (see spans.go).
//
//   - Noul: one logit z = w.x + b;
//   - Choice: K logits z = W.x + b;
//   - Score: K-1 ordinal logits z_k = w_k.x + b_k, with P(level > k) =
//     sigma(z_k) (Frank and Hall decomposition). Each threshold has its own
//     vector: with a shared vector (CORAL), separating an intermediate level
//     requires pulling the biases apart, which Adam does at a rate of
//     about lr per step. Consistency between thresholds is restored on
//     read (see answer).
type head struct {
	q    Question
	w    []float32 // [rows, H]
	b    []float32 // [outs]
	gw   []float32
	gb   []float32
	rows int // rows of w: 1, or K for Choice
	outs int // number of logits
}

func newHead(q Question, hidden int, rng *rand.Rand) *head {
	h := &head{q: q}
	switch q.Kind {
	case Noul:
		h.rows, h.outs = 1, 1
	case Choice:
		h.rows, h.outs = len(q.Options), len(q.Options)
	case Score:
		h.rows, h.outs = len(q.Options)-1, len(q.Options)-1
	case Spans:
		h.rows = spanTagCount(len(q.Options))
		h.outs = h.rows
	}
	h.w = make([]float32, h.rows*hidden)
	h.b = make([]float32, h.outs)
	for i := range h.w {
		h.w[i] = float32(rng.NormFloat64() * 0.02)
	}
	return h
}

func (h *head) enableGrad() {
	if h.gw == nil {
		h.gw = make([]float32, len(h.w))
		h.gb = make([]float32, len(h.b))
	}
}

func (h *head) zeroGrad() {
	clear(h.gw)
	clear(h.gb)
}

// logits computes the logits of a pooled vector x.
func (h *head) logits(x []float32) []float64 {
	H := len(x)
	z := make([]float64, h.outs)
	for r := 0; r < h.rows; r++ {
		var s float64
		for i, v := range h.w[r*H : (r+1)*H] {
			s += float64(v) * float64(x[i])
		}
		z[r] = s + float64(h.b[r])
	}
	return z
}

// lossGrad returns the loss for target t (see Question.target), and the
// gradient of the loss with respect to the logits.
func (h *head) lossGrad(z, t []float64) (float64, []float64) {
	dz := make([]float64, len(z))
	var loss float64
	switch h.q.Kind {
	case Noul:
		// BCE with logits: softplus(z) - y*z
		loss = softplus(z[0]) - t[0]*z[0]
		dz[0] = sigmoid(z[0]) - t[0]
	case Choice:
		p := softmax(z)
		for i := range z {
			if t[i] > 0 {
				loss -= t[i] * math.Log(max(p[i], 1e-300))
			}
			dz[i] = p[i] - t[i]
		}
	case Score:
		level := int(t[0])
		for k := range z {
			y := 0.0
			if level > k {
				y = 1
			}
			loss += softplus(z[k]) - y*z[k]
			dz[k] = sigmoid(z[k]) - y
		}
	}
	return loss, dz
}

// backward accumulates the weight gradients and returns dL/dx, for a
// gradient dz on the logits of input x.
func (h *head) backward(x []float32, dz []float64, dx []float32) {
	H := len(x)
	for r := 0; r < h.rows; r++ {
		d := dz[r]
		h.gb[r] += float32(d)
		w := h.w[r*H : (r+1)*H]
		gw := h.gw[r*H : (r+1)*H]
		for i := range x {
			gw[i] += float32(d * float64(x[i]))
			dx[i] += float32(d * float64(w[i]))
		}
	}
}

// answer turns the logits, divided by the temperature, into an answer.
func (h *head) answer(z []float64, temperature float64) Answer {
	if temperature <= 0 {
		temperature = 1
	}
	zt := make([]float64, len(z))
	for i, v := range z {
		zt[i] = v / temperature
	}
	a := Answer{Question: h.q.Name, Kind: h.q.Kind}
	switch h.q.Kind {
	case Noul:
		a.P = sigmoid(zt[0])
		a.Confidence = max(a.P, 1-a.P)
	case Choice:
		p := softmax(zt)
		a.Probs = make(map[string]float64, len(p))
		best := 0
		for i, v := range p {
			a.Probs[h.q.Options[i]] = v
			if v > p[best] {
				best = i
			}
		}
		a.Choice = h.q.Options[best]
		a.Confidence = p[best]
		a.Margin = margin(p, best)
	case Score:
		// P(level > k), made monotone: an ordinal model can produce
		// slight inversions between thresholds.
		gt := make([]float64, len(zt))
		for k, v := range zt {
			gt[k] = sigmoid(v)
			if k > 0 {
				gt[k] = min(gt[k], gt[k-1])
			}
		}
		levels := len(h.q.Options)
		dist := make([]float64, levels)
		for k := 0; k < levels; k++ {
			above := 1.0
			if k > 0 {
				above = gt[k-1]
			}
			below := 0.0
			if k < len(gt) {
				below = gt[k]
			}
			dist[k] = above - below
		}
		a.Probs = make(map[string]float64, levels)
		best := 0
		for k, v := range dist {
			a.Probs[h.q.Options[k]] = v
			a.Score += float64(k) * v
			if v > dist[best] {
				best = k
			}
		}
		a.Choice = h.q.Options[best]
		a.Confidence = dist[best]
	}
	return a
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}

// softplus(z) = log(1 + e^z), stable for large |z|.
func softplus(z float64) float64 {
	if z > 30 {
		return z
	}
	if z < -30 {
		return math.Exp(z)
	}
	return math.Log1p(math.Exp(z))
}

func softmax(z []float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range z {
		mx = max(mx, v)
	}
	p := make([]float64, len(z))
	var sum float64
	for i, v := range z {
		p[i] = math.Exp(v - mx)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// margin returns p[best] minus the average of the other probabilities.
func margin(p []float64, best int) float64 {
	if len(p) < 2 {
		return p[best]
	}
	var others float64
	for i, v := range p {
		if i != best {
			others += v
		}
	}
	return p[best] - others/float64(len(p)-1)
}
