package teacher

import (
	"fmt"
	"sort"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// Verdict is a teacher's opinion on an example.
type Verdict struct {
	Teacher string         `json:"teacher"`
	Labels  map[string]any `json:"labels"`
}

// Disagreement is an example on which the teachers, or a teacher and the
// existing label, contradict each other: it needs a human review.
type Disagreement struct {
	Example  dataset.Example `json:"example"`
	Verdicts []Verdict       `json:"verdicts"`
	// Reason names the question at issue.
	Reason string `json:"reason"`
}

// Consensus merges the labels of several teachers on the same examples.
// byTeacher[t][i] is teacher t's opinion on examples[i] (nil if it did not
// answer).
//
// A single teacher can be confidently wrong; several different teachers
// rarely make the same mistake. So we keep what they agree on, and hand the
// rest to a human:
//   - Noul: all on the same side of 0.5; the kept probability is the
//     average. Otherwise the example goes to disagreement.
//   - Choice: same option chosen by all; average distribution. Otherwise
//     the question is removed from the example, which stays usable.
//   - Score: same level for all, otherwise the question is removed.
//
// A label already present on the example (exact, by construction or by
// provenance) is kept; if the teachers unanimously contradict it on a noul
// question, the example goes to disagreement: it is often the source that
// is wrong, sometimes the teachers, and only a human can decide.
func Consensus(schema indecis.Schema, examples []dataset.Example, names []string, byTeacher [][]map[string]any) ([]dataset.Example, []Disagreement) {
	var kept []dataset.Example
	var disputes []Disagreement
	for i, e := range examples {
		var verdicts []Verdict
		for t := range byTeacher {
			if byTeacher[t][i] != nil {
				verdicts = append(verdicts, Verdict{Teacher: names[t], Labels: byTeacher[t][i]})
			}
		}
		if len(verdicts) < len(byTeacher) {
			continue // a teacher did not answer: no consensus possible
		}
		labels := map[string]any{}
		reason := ""
		for _, q := range schema {
			switch q.Kind {
			case indecis.Noul:
				ps := make([]float64, 0, len(verdicts))
				for _, v := range verdicts {
					if p, ok := v.Labels[q.Name].(float64); ok {
						ps = append(ps, p)
					}
				}
				if len(ps) < len(verdicts) {
					continue
				}
				yes := 0
				var mean float64
				for _, p := range ps {
					mean += p / float64(len(ps))
					if p >= 0.5 {
						yes++
					}
				}
				if yes != 0 && yes != len(ps) {
					reason = fmt.Sprintf("%s: the teachers disagree", q.Name)
					break
				}
				if exact, ok := e.Labels[q.Name].(bool); ok && exact != (mean >= 0.5) {
					reason = fmt.Sprintf("%s: the teachers contradict the existing label (%v)", q.Name, exact)
					break
				}
				labels[q.Name] = mean
			case indecis.Choice:
				if d, ok := agreedChoice(q, verdicts); ok {
					labels[q.Name] = d
				}
			case indecis.Score:
				if l, ok := agreedLevel(q, verdicts); ok {
					labels[q.Name] = l
				}
			}
			if reason != "" {
				break
			}
		}
		if reason != "" {
			disputes = append(disputes, Disagreement{Example: e, Verdicts: verdicts, Reason: reason})
			continue
		}
		out := e
		out.Labels = mergeLabels(e.Labels, labels)
		out.Meta = withMeta(e.Meta, "teachers", joinNames(names))
		kept = append(kept, out)
	}
	return kept, disputes
}

// agreedChoice returns the average distribution if all teachers pick the
// same option.
func agreedChoice(q indecis.Question, verdicts []Verdict) (map[string]any, bool) {
	avg := map[string]float64{}
	best := ""
	for _, v := range verdicts {
		d, ok := v.Labels[q.Name].(map[string]any)
		if !ok {
			return nil, false
		}
		top, topP := "", -1.0
		for o, pv := range d {
			p, _ := pv.(float64)
			avg[o] += p / float64(len(verdicts))
			if p > topP || (p == topP && o < top) {
				top, topP = o, p
			}
		}
		if best == "" {
			best = top
		} else if best != top {
			return nil, false
		}
	}
	out := map[string]any{}
	for o, p := range avg {
		out[o] = p
	}
	return out, true
}

func agreedLevel(q indecis.Question, verdicts []Verdict) (string, bool) {
	level := ""
	for _, v := range verdicts {
		l, ok := v.Labels[q.Name].(string)
		if !ok || (level != "" && l != level) {
			return "", false
		}
		level = l
	}
	return level, level != ""
}

func joinNames(names []string) string {
	s := append([]string(nil), names...)
	sort.Strings(s)
	out := ""
	for i, n := range s {
		if i > 0 {
			out += ","
		}
		out += n
	}
	return out
}
