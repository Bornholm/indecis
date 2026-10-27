package ocr

import (
	"math"
	"sort"
	"strings"

	"github.com/bornholm/indecis/internal/charlm"
	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linerec"
)

// LMOptions sets the decoding of uncertain lines with a character language
// model: a CTC prefix beam search whose score adds Alpha·log P_lm of each
// new character and Beta per character (against the model's taste for
// short texts).
type LMOptions struct {
	Alpha float64 // weight of the language model (0.3)
	Beta  float64 // bonus per character (1.0)
	Beam  int     // prefixes kept per step (8)
	Top   int     // characters tried per step (6)
	// Gate: a line whose greedy reading is at least this sure is kept as
	// is (0.9): the search costs time and only helps where the recognizer
	// hesitates.
	Gate float32
}

// DefaultLM returns the default decoding options.
func DefaultLM() LMOptions { return LMOptions{Alpha: 0.3, Beta: 1.0, Beam: 8, Top: 6, Gate: 0.9} }

// WithLanguageModel decodes uncertain lines with a character language
// model (see LMOptions).
func WithLanguageModel(lm *charlm.Model, o LMOptions) Option {
	return func(opts *options) { opts.lm, opts.lmOpts = lm, o }
}

type prefix struct {
	ids     []int  // symbols
	runes   []rune // their text, for the language model
	steps   []int  // the step each symbol was first emitted at
	pb, pnb float64
	lm      float64 // Alpha·Σ log P_lm + Beta·characters
}

func (p *prefix) total() float64 { return logAdd(p.pb, p.pnb) }

func (p *prefix) key() string {
	var b strings.Builder
	for _, id := range p.ids {
		b.WriteRune(rune(id))
	}
	return b.String()
}

// beamDecode reads logits [L, V] with a CTC prefix beam search guided by
// lm. chars[i] is symbol i+1's text.
func beamDecode(logits []float32, L, V int, chars []string, lm *charlm.Model, o LMOptions) []linerec.Symbol {
	linalg.ZeroUpper()
	inf := math.Inf(-1)
	lp := make([]float64, V)
	beams := []*prefix{{pb: 0, pnb: inf}}
	order := make([]int, V)
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
			lp[k] = float64(v) - lse
		}
		// A step that is almost surely blank only adds blank.
		if lp[0] > math.Log(0.999) {
			for _, b := range beams {
				b.pb, b.pnb = b.total()+lp[0], inf
			}
			continue
		}
		for k := range order {
			order[k] = k
		}
		sort.Slice(order, func(a, b int) bool { return lp[order[a]] > lp[order[b]] })
		cands := order[:min(o.Top+1, V)]

		next := map[string]*prefix{}
		get := func(from *prefix, add int) *prefix {
			ids := from.ids
			if add > 0 {
				ids = append(append([]int(nil), from.ids...), add)
			}
			p := &prefix{ids: ids}
			k := p.key()
			if q, ok := next[k]; ok {
				return q
			}
			p.pb, p.pnb = inf, inf
			p.runes, p.steps, p.lm = from.runes, from.steps, from.lm
			if add > 0 {
				p.steps = append(append([]int(nil), from.steps...), t)
				p.runes = append([]rune(nil), from.runes...)
				for _, r := range chars[add-1] {
					p.lm += o.Alpha*math.Log(lm.Prob(p.runes, r)) + o.Beta
					p.runes = append(p.runes, r)
				}
			}
			next[k] = p
			return p
		}
		for _, b := range beams {
			total := b.total()
			same := get(b, 0)
			same.pb = logAdd(same.pb, total+lp[0])
			last := 0
			if len(b.ids) > 0 {
				last = b.ids[len(b.ids)-1]
			}
			tried := false
			for _, c := range cands {
				if c == 0 {
					continue
				}
				if c == last {
					tried = true
					same.pnb = logAdd(same.pnb, b.pnb+lp[c])
					ext := get(b, c)
					ext.pnb = logAdd(ext.pnb, b.pb+lp[c])
					continue
				}
				ext := get(b, c)
				ext.pnb = logAdd(ext.pnb, total+lp[c])
			}
			if last > 0 && !tried {
				// The last symbol held over a step stays possible even
				// outside the step's best candidates.
				same.pnb = logAdd(same.pnb, b.pnb+lp[last])
			}
		}
		beams = beams[:0]
		for _, p := range next {
			beams = append(beams, p)
		}
		sort.Slice(beams, func(a, b int) bool {
			return beams[a].total()+beams[a].lm > beams[b].total()+beams[b].lm
		})
		if len(beams) > o.Beam {
			beams = beams[:o.Beam]
		}
	}
	best := beams[0]
	syms := make([]linerec.Symbol, len(best.ids))
	for i, id := range best.ids {
		end := L
		if i+1 < len(best.steps) {
			end = best.steps[i+1]
		}
		syms[i] = linerec.Symbol{ID: id, Step: best.steps[i], Steps: max(1, end-best.steps[i])}
	}
	return syms
}

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
