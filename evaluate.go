package indecis

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/bornholm/indecis/dataset"
)

// Metrics measures a question over a set of examples.
type Metrics struct {
	Question string `json:"question"`
	Kind     Kind   `json:"kind"`
	N        int    `json:"n"`
	// Accuracy: most probable answer = label. For Noul, threshold 0.5;
	// for a soft label, it is its rounding.
	Accuracy float64 `json:"accuracy"`
	// Noul: precision, recall and F1 of the "true" class, area under the
	// ROC curve.
	Precision float64 `json:"precision,omitempty"`
	Recall    float64 `json:"recall,omitempty"`
	F1        float64 `json:"f1,omitempty"`
	AUC       float64 `json:"auc,omitempty"`
	// Choice: average F1 over the options present.
	MacroF1 float64 `json:"macro_f1,omitempty"`
	// Score: mean absolute deviation between expected level and actual level.
	MAE float64 `json:"mae,omitempty"`
	// Spans: Precision, Recall and F1 count the passages found with their
	// exact type and bounds, F2 weighs recall twice, Types details each
	// type. Accuracy, NLL and ECE are per token.
	F2    float64       `json:"f2,omitempty"`
	Types []SpanMetrics `json:"types,omitempty"`
	// NLL is the mean negative log-likelihood, Brier the Brier score
	// (Noul), ECE the calibration error over 10 confidence bins.
	NLL   float64 `json:"nll"`
	Brier float64 `json:"brier,omitempty"`
	ECE   float64 `json:"ece"`
}

// Evaluate measures the model, including temperatures.
func (m *Model) Evaluate(ctx context.Context, examples []dataset.Example) ([]Metrics, error) {
	data, err := m.encode(examples)
	if err != nil {
		return nil, err
	}
	res, err := m.infer(ctx, examplesToInputs(examples), 32)
	if err != nil {
		return nil, err
	}
	logits := make([][][]float64, len(res))
	texts := make([]string, len(examples))
	for i, r := range res {
		logits[i] = r.pooled
		texts[i] = examples[i].Text
	}
	var out []Metrics
	for qi, h := range m.heads {
		if h.q.Kind == Spans {
			out = append(out, m.spanMetrics(qi, data, texts, res))
			continue
		}
		var answers []Answer
		var targets [][]float64
		var zs [][]float64
		for i, e := range data {
			if e.targets[qi] == nil {
				continue
			}
			answers = append(answers, h.answer(logits[i][qi], m.temps[qi]))
			targets = append(targets, e.targets[qi])
			z := make([]float64, len(logits[i][qi]))
			for k, v := range logits[i][qi] {
				z[k] = v / m.temps[qi]
			}
			zs = append(zs, z)
		}
		out = append(out, h.metrics(answers, targets, zs))
	}
	return out, nil
}

func (h *head) metrics(answers []Answer, targets, zs [][]float64) Metrics {
	mt := Metrics{Question: h.q.Name, Kind: h.q.Kind, N: len(answers)}
	if mt.N == 0 {
		return mt
	}
	var conf []float64
	var correct []bool
	for i, a := range answers {
		l, _ := h.lossGrad(zs[i], targets[i])
		mt.NLL += l
		t := targets[i]
		switch h.q.Kind {
		case Noul:
			y := t[0] >= 0.5
			pred := a.P >= 0.5
			correct = append(correct, y == pred)
			conf = append(conf, a.Confidence)
			mt.Brier += (a.P - t[0]) * (a.P - t[0])
		case Choice:
			ok := t[indexOf(h.q.Options, a.Choice)] == maxOf(t)
			correct = append(correct, ok)
			conf = append(conf, a.Confidence)
		case Score:
			level := int(t[0])
			correct = append(correct, a.Choice == h.q.Options[level])
			conf = append(conf, a.Confidence)
			mt.MAE += math.Abs(a.Score - t[0])
		}
	}
	n := float64(mt.N)
	mt.NLL /= n
	mt.Brier /= n
	mt.MAE /= n
	for _, c := range correct {
		if c {
			mt.Accuracy++
		}
	}
	mt.Accuracy /= n
	mt.ECE = ece(conf, correct, 10)

	switch h.q.Kind {
	case Noul:
		var tp, fp, fn float64
		scores := make([]float64, len(answers))
		labels := make([]bool, len(answers))
		for i, a := range answers {
			y, pred := targets[i][0] >= 0.5, a.P >= 0.5
			switch {
			case y && pred:
				tp++
			case !y && pred:
				fp++
			case y && !pred:
				fn++
			}
			// Ranking is done on the logit: the probability saturates to
			// 1 in float64 as soon as z > 37, and ties would skew the AUC.
			scores[i], labels[i] = zs[i][0], y
		}
		mt.Precision = safeDiv(tp, tp+fp)
		mt.Recall = safeDiv(tp, tp+fn)
		mt.F1 = safeDiv(2*mt.Precision*mt.Recall, mt.Precision+mt.Recall)
		mt.AUC = auc(scores, labels)
	case Choice:
		var f1s []float64
		for oi, o := range h.q.Options {
			var tp, fp, fn float64
			present := false
			for i, a := range answers {
				y := targets[i][oi] == maxOf(targets[i])
				present = present || y
				pred := a.Choice == o
				switch {
				case y && pred:
					tp++
				case !y && pred:
					fp++
				case y && !pred:
					fn++
				}
			}
			if present {
				p, r := safeDiv(tp, tp+fp), safeDiv(tp, tp+fn)
				f1s = append(f1s, safeDiv(2*p*r, p+r))
			}
		}
		for _, f := range f1s {
			mt.MacroF1 += f
		}
		mt.MacroF1 = safeDiv(mt.MacroF1, float64(len(f1s)))
	}
	return mt
}

// ece is the expected calibration error: the average gap, weighted by
// count, between confidence and accuracy in each confidence bin.
func ece(conf []float64, correct []bool, bins int) float64 {
	sumC := make([]float64, bins)
	sumA := make([]float64, bins)
	count := make([]float64, bins)
	for i, c := range conf {
		b := min(int(c*float64(bins)), bins-1)
		sumC[b] += c
		if correct[i] {
			sumA[b]++
		}
		count[b]++
	}
	var e float64
	for b := range count {
		if count[b] > 0 {
			e += math.Abs(sumC[b]-sumA[b]) / float64(len(conf))
		}
	}
	return e
}

// auc computes the area under the ROC curve by ranks (Mann-Whitney), ties
// counted for half.
func auc(scores []float64, labels []bool) float64 {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return scores[idx[a]] < scores[idx[b]] })
	var pos, neg, rankSum float64
	for i := 0; i < len(idx); {
		j := i
		for j < len(idx) && scores[idx[j]] == scores[idx[i]] {
			j++
		}
		avgRank := float64(i+j+1) / 2
		for k := i; k < j; k++ {
			if labels[idx[k]] {
				rankSum += avgRank
				pos++
			} else {
				neg++
			}
		}
		i = j
	}
	if pos == 0 || neg == 0 {
		return math.NaN()
	}
	return (rankSum - pos*(pos+1)/2) / (pos * neg)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func maxOf(v []float64) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = max(m, x)
	}
	return m
}

// String presents the metrics on one line.
func (mt Metrics) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-12s n=%-5d acc=%.3f", mt.Question, mt.N, mt.Accuracy)
	switch mt.Kind {
	case Noul:
		fmt.Fprintf(&b, " P=%.3f R=%.3f F1=%.3f AUC=%.3f Brier=%.4f", mt.Precision, mt.Recall, mt.F1, mt.AUC, mt.Brier)
	case Choice:
		fmt.Fprintf(&b, " macroF1=%.3f", mt.MacroF1)
	case Score:
		fmt.Fprintf(&b, " MAE=%.3f", mt.MAE)
	case Spans:
		fmt.Fprintf(&b, " P=%.3f R=%.3f F1=%.3f F2=%.3f", mt.Precision, mt.Recall, mt.F1, mt.F2)
	}
	fmt.Fprintf(&b, " NLL=%.4f ECE=%.4f", mt.NLL, mt.ECE)
	for _, s := range mt.Types {
		fmt.Fprintf(&b, "\n  %-10s gold=%-6d found=%-6d P=%.3f R=%.3f F1=%.3f F2=%.3f", s.Type, s.Gold, s.Found, s.Precision, s.Recall, s.F1, s.F2)
	}
	return b.String()
}
