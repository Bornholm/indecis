package teacher

import (
	"fmt"
	"sort"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// Verdict est l'avis d'un teacher sur un exemple.
type Verdict struct {
	Teacher string         `json:"teacher"`
	Labels  map[string]any `json:"labels"`
}

// Disagreement est un exemple sur lequel les teachers, ou un teacher et
// l'étiquette existante, se contredisent : il est à relire.
type Disagreement struct {
	Example  dataset.Example `json:"example"`
	Verdicts []Verdict       `json:"verdicts"`
	// Reason nomme la question en cause.
	Reason string `json:"reason"`
}

// Consensus fusionne les étiquettes de plusieurs teachers sur les mêmes
// exemples. byTeacher[t][i] est l'avis du teacher t sur examples[i] (nil
// s'il n'a pas répondu).
//
// Un seul teacher se trompe avec assurance ; plusieurs teachers différents se
// trompent rarement de la même façon. On garde donc ce sur quoi ils
// s'accordent, et on renvoie le reste à un humain :
//   - Noul : tous du même côté de 0,5 ; la probabilité retenue est la
//     moyenne. Sinon l'exemple part en désaccord.
//   - Choice : même option retenue par tous ; distribution moyenne. Sinon la
//     question est retirée de l'exemple, qui reste utilisable.
//   - Score : même niveau pour tous, sinon la question est retirée.
//
// Une étiquette déjà présente sur l'exemple (exacte, par construction ou par
// provenance) est conservée ; si les teachers la contredisent unanimement
// sur une question noul, l'exemple part en désaccord : c'est souvent la
// source qui se trompe, parfois les teachers, et seul un humain tranche.
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
			continue // un teacher n'a pas répondu : pas de consensus possible
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
					reason = fmt.Sprintf("%s : les teachers divergent", q.Name)
					break
				}
				if exact, ok := e.Labels[q.Name].(bool); ok && exact != (mean >= 0.5) {
					reason = fmt.Sprintf("%s : les teachers contredisent l'étiquette existante (%v)", q.Name, exact)
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

// agreedChoice retourne la distribution moyenne si tous les teachers
// retiennent la même option.
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
