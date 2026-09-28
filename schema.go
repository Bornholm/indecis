// Package indecis builds small decision models: a pretrained text encoder,
// fully fine-tuned in pure Go, that answers typed questions with calibrated
// probabilities instead of generated text.
//
// A [Schema] lists the questions, of three types:
//   - noul ([NewNoul]): a yes/no question, answered with P(yes);
//   - choice ([NewChoice]): one option among several, answered with a
//     distribution;
//   - score ([NewScore]): a level on an ordered scale, answered with the
//     expected level and its distribution.
//
// One pass of the encoder answers every question of a schema: each question
// is one more head on the same representation of the text.
//
//	m, _ := indecis.New(backboneDir, schema, 1)
//	_ = m.Fit(ctx, train, indecis.DefaultTrainOptions())
//	_, _ = m.Calibrate(ctx, calib)
//	_ = m.Save("model")
//
//	m, _ = indecis.Load("model", indecis.WithInt8())
//	d, _ := m.Decide(ctx, "My parcel has not arrived")
//
// In open mode, the options are described in text at call time and compared
// with the text through embeddings ([Model.ChooseNearest],
// [Model.PrepareCandidates], [Model.DecideOpen]); [Model.FitEmbeddings]
// fine-tunes the encoder for it.
//
// The guides in the docs directory of the repository cover the whole
// process, from the schema to the deployment.
package indecis

import (
	"fmt"
	"regexp"
)

// Kind est le type d'une question.
type Kind string

const (
	Noul   Kind = "noul"
	Choice Kind = "choice"
	Score  Kind = "score"
)

// Question décrit une décision que le modèle apprend à prendre.
type Question struct {
	Name string `json:"name"`
	Kind Kind   `json:"kind"`
	// Instructions documente la question. Le modèle ne la lit pas : ses têtes
	// sont apprises, la question est dans les données.
	Instructions string `json:"instructions,omitempty"`
	// Options liste les options d'un Choice, ou les niveaux d'un Score dans
	// l'ordre croissant.
	Options []string `json:"options,omitempty"`
}

// Schema est la liste des questions d'un modèle.
type Schema []Question

// NewNoul, NewChoice et NewScore construisent les questions.
func NewNoul(name, instructions string) Question {
	return Question{Name: name, Kind: Noul, Instructions: instructions}
}

func NewChoice(name, instructions string, options ...string) Question {
	return Question{Name: name, Kind: Choice, Instructions: instructions, Options: options}
}

func NewScore(name, instructions string, levels ...string) Question {
	return Question{Name: name, Kind: Score, Instructions: instructions, Options: levels}
}

var nameRe = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// Validate vérifie le schéma.
func (s Schema) Validate() error {
	if len(s) == 0 {
		return fmt.Errorf("indecis: schéma vide")
	}
	seen := map[string]bool{}
	for _, q := range s {
		if !nameRe.MatchString(q.Name) {
			return fmt.Errorf("indecis: nom de question %q invalide (attendu [a-z][a-z0-9_]*)", q.Name)
		}
		if seen[q.Name] {
			return fmt.Errorf("indecis: question %q en double", q.Name)
		}
		seen[q.Name] = true
		switch q.Kind {
		case Noul:
			if len(q.Options) != 0 {
				return fmt.Errorf("indecis: %s : une question noul n'a pas d'options", q.Name)
			}
		case Choice, Score:
			if len(q.Options) < 2 {
				return fmt.Errorf("indecis: %s : au moins deux options", q.Name)
			}
			opts := map[string]bool{}
			for _, o := range q.Options {
				if o == "" || opts[o] {
					return fmt.Errorf("indecis: %s : option %q vide ou en double", q.Name, o)
				}
				opts[o] = true
			}
		default:
			return fmt.Errorf("indecis: %s : type %q inconnu", q.Name, q.Kind)
		}
	}
	return nil
}

// Index retourne la position d'une question, -1 si absente.
func (s Schema) Index(name string) int {
	for i, q := range s {
		if q.Name == name {
			return i
		}
	}
	return -1
}

// target convertit une étiquette de dataset en cible d'entraînement :
// une probabilité pour Noul, une distribution pour Choice, un niveau pour
// Score. ok vaut false si l'étiquette est absente.
func (q Question) target(v any) (t []float64, ok bool, err error) {
	if v == nil {
		return nil, false, nil
	}
	switch q.Kind {
	case Noul:
		switch x := v.(type) {
		case bool:
			if x {
				return []float64{1}, true, nil
			}
			return []float64{0}, true, nil
		case float64:
			if x < 0 || x > 1 {
				return nil, false, fmt.Errorf("%s : probabilité %v hors de [0, 1]", q.Name, x)
			}
			return []float64{x}, true, nil
		}
	case Choice:
		switch x := v.(type) {
		case string:
			t = make([]float64, len(q.Options))
			for i, o := range q.Options {
				if o == x {
					t[i] = 1
					return t, true, nil
				}
			}
			return nil, false, fmt.Errorf("%s : option %q inconnue", q.Name, x)
		case map[string]any:
			t = make([]float64, len(q.Options))
			var sum float64
			for k, pv := range x {
				p, isNum := pv.(float64)
				i := indexOf(q.Options, k)
				if !isNum || i < 0 || p < 0 {
					return nil, false, fmt.Errorf("%s : distribution invalide sur %q", q.Name, k)
				}
				t[i] = p
				sum += p
			}
			if sum <= 0 {
				return nil, false, fmt.Errorf("%s : distribution vide", q.Name)
			}
			for i := range t {
				t[i] /= sum
			}
			return t, true, nil
		}
	case Score:
		if x, isNum := v.(float64); isNum {
			if x < 0 || x > float64(len(q.Options)-1) || x != float64(int(x)) {
				return nil, false, fmt.Errorf("%s : niveau %v invalide", q.Name, x)
			}
			return []float64{x}, true, nil
		}
		if x, isStr := v.(string); isStr {
			if i := indexOf(q.Options, x); i >= 0 {
				return []float64{float64(i)}, true, nil
			}
			return nil, false, fmt.Errorf("%s : niveau %q inconnu", q.Name, x)
		}
	}
	return nil, false, fmt.Errorf("%s : étiquette %v (%T) incompatible avec le type %s", q.Name, v, v, q.Kind)
}

func indexOf(s []string, v string) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return -1
}
