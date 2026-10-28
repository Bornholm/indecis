package linerec

import (
	"math"

	"github.com/bornholm/indecis/linalg"
)

// CTC computes the connectionist temporal classification loss of a line,
// -log p(target | logits), and its gradient with respect to the logits
// (written into dLogits, [L, V], when not nil): softmax minus the
// posterior occupancy of each symbol. Blank is symbol 0; target holds
// symbols 1 to V-1. ok is false when no alignment fits in L steps (too
// short a line for its text): the loss is then +Inf and dLogits is left
// at zero, as PyTorch's zero_infinity.
//
// The forward (α) and backward (β) recursions run in log space, in
// float64.
func CTC(logits []float32, L, V int, target []int, dLogits []float32) (loss float64, ok bool) {
	linalg.ZeroUpper() // the recursions below are scalar float64
	S := 2*len(target) + 1
	label := func(s int) int {
		if s%2 == 0 {
			return 0
		}
		return target[s/2]
	}
	// Log-softmax of each step.
	lp := make([]float64, L*V)
	for t := range L {
		row := logits[t*V : (t+1)*V]
		mx := float64(row[0])
		for _, v := range row {
			mx = max(mx, float64(v))
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v) - mx)
		}
		lse := mx + math.Log(sum)
		for k, v := range row {
			lp[t*V+k] = float64(v) - lse
		}
	}

	inf := math.Inf(-1)
	alpha := make([]float64, L*S)
	beta := make([]float64, L*S)
	for i := range alpha {
		alpha[i], beta[i] = inf, inf
	}
	// skip reports whether s may come from s-2 (a label different from the
	// one before the blank between them).
	skip := func(s int) bool { return s >= 2 && s%2 == 1 && label(s) != label(s-2) }

	alpha[0] = lp[0]
	if S > 1 {
		alpha[1] = lp[label(1)]
	}
	for t := 1; t < L; t++ {
		for s := range S {
			a := alpha[(t-1)*S+s]
			if s >= 1 {
				a = logAdd(a, alpha[(t-1)*S+s-1])
			}
			if skip(s) {
				a = logAdd(a, alpha[(t-1)*S+s-2])
			}
			alpha[t*S+s] = a + lp[t*V+label(s)]
		}
	}
	logP := alpha[(L-1)*S+S-1]
	if S > 1 {
		logP = logAdd(logP, alpha[(L-1)*S+S-2])
	}
	if math.IsInf(logP, -1) {
		if dLogits != nil {
			clear(dLogits[:L*V])
		}
		return math.Inf(1), false
	}
	if dLogits == nil {
		return -logP, true
	}

	beta[(L-1)*S+S-1] = lp[(L-1)*V]
	if S > 1 {
		beta[(L-1)*S+S-2] = lp[(L-1)*V+label(S-2)]
	}
	for t := L - 2; t >= 0; t-- {
		for s := range S {
			b := beta[(t+1)*S+s]
			if s+1 < S {
				b = logAdd(b, beta[(t+1)*S+s+1])
			}
			if s+2 < S && skip(s+2) {
				b = logAdd(b, beta[(t+1)*S+s+2])
			}
			beta[t*S+s] = b + lp[t*V+label(s)]
		}
	}

	// dz[t,k] = softmax[t,k] − Σ_{s: label(s)=k} exp(α+β − lp[t,k] − logP).
	occ := make([]float64, V)
	for t := range L {
		clear(occ)
		for s := range S {
			k := label(s)
			v := alpha[t*S+s] + beta[t*S+s] - lp[t*V+k] - logP
			if !math.IsInf(v, -1) {
				occ[k] += math.Exp(v)
			}
		}
		d := dLogits[t*V : (t+1)*V]
		for k := range V {
			d[k] = float32(math.Exp(lp[t*V+k]) - occ[k])
		}
	}
	return -logP, true
}

// logAdd returns log(e^a + e^b).
func logAdd(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}
