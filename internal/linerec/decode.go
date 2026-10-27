package linerec

import (
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// Symbol is a symbol read by greedy CTC decoding.
type Symbol struct {
	ID    int     // vocabulary index (blank = 0 never appears)
	Step  int     // first step of its run
	Steps int     // length of its run
	P     float32 // highest probability over its run
}

// Greedy decodes logits [L, V] along the best path: the most likely symbol
// of each step, repeats collapsed, blanks removed. confidence is the
// lowest probability among the chosen symbols, blanks included: a line's
// weakest step.
func Greedy(logits []float32, L, V int) (syms []Symbol, confidence float32) {
	linalg.ZeroUpper()
	confidence = 1
	prev := 0
	for t := range L {
		row := logits[t*V : (t+1)*V]
		best, mx := 0, row[0]
		for i, v := range row {
			if v > mx {
				best, mx = i, v
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - mx))
		}
		p := float32(1 / sum)
		confidence = min(confidence, p)
		switch {
		case best == 0:
		case best == prev:
			last := &syms[len(syms)-1]
			last.Steps++
			last.P = max(last.P, p)
		default:
			syms = append(syms, Symbol{ID: best, Step: t, Steps: 1, P: p})
		}
		prev = best
	}
	return syms, confidence
}
