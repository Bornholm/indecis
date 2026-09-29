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

// Kind is the type of a question.
type Kind string

const (
	Noul   Kind = "noul"
	Choice Kind = "choice"
	Score  Kind = "score"
)

// Question describes a decision that the model learns to make.
type Question struct {
	Name string `json:"name"`
	Kind Kind   `json:"kind"`
	// Instructions documents the question. The model does not read it: its
	// heads are learned, the question is in the data.
	Instructions string `json:"instructions,omitempty"`
	// Options lists the options of a Choice, or the levels of a Score in
	// increasing order.
	Options []string `json:"options,omitempty"`
}

// Schema is the list of a model's questions.
type Schema []Question

// NewNoul, NewChoice and NewScore build the questions.
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

// Validate checks the schema.
func (s Schema) Validate() error {
	if len(s) == 0 {
		return fmt.Errorf("indecis: empty schema")
	}
	seen := map[string]bool{}
	for _, q := range s {
		if !nameRe.MatchString(q.Name) {
			return fmt.Errorf("indecis: invalid question name %q (expected [a-z][a-z0-9_]*)", q.Name)
		}
		if seen[q.Name] {
			return fmt.Errorf("indecis: duplicate question %q", q.Name)
		}
		seen[q.Name] = true
		switch q.Kind {
		case Noul:
			if len(q.Options) != 0 {
				return fmt.Errorf("indecis: %s: a noul question has no options", q.Name)
			}
		case Choice, Score:
			if len(q.Options) < 2 {
				return fmt.Errorf("indecis: %s: at least two options", q.Name)
			}
			opts := map[string]bool{}
			for _, o := range q.Options {
				if o == "" || opts[o] {
					return fmt.Errorf("indecis: %s: option %q empty or duplicate", q.Name, o)
				}
				opts[o] = true
			}
		default:
			return fmt.Errorf("indecis: %s: unknown type %q", q.Name, q.Kind)
		}
	}
	return nil
}

// Index returns the position of a question, -1 if absent.
func (s Schema) Index(name string) int {
	for i, q := range s {
		if q.Name == name {
			return i
		}
	}
	return -1
}

// Target converts a dataset label into a training target: a probability
// for Noul, a distribution for Choice, a level for Score. ok is false if
// the label is absent. For heads trained outside this package (vision).
func (q Question) Target(v any) (t []float64, ok bool, err error) { return q.target(v) }

// target converts a dataset label into a training target: a probability
// for Noul, a distribution for Choice, a level for Score. ok is false if
// the label is absent.
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
				return nil, false, fmt.Errorf("%s: probability %v out of [0, 1]", q.Name, x)
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
			return nil, false, fmt.Errorf("%s: unknown option %q", q.Name, x)
		case map[string]any:
			t = make([]float64, len(q.Options))
			var sum float64
			for k, pv := range x {
				p, isNum := pv.(float64)
				i := indexOf(q.Options, k)
				if !isNum || i < 0 || p < 0 {
					return nil, false, fmt.Errorf("%s: invalid distribution on %q", q.Name, k)
				}
				t[i] = p
				sum += p
			}
			if sum <= 0 {
				return nil, false, fmt.Errorf("%s: empty distribution", q.Name)
			}
			for i := range t {
				t[i] /= sum
			}
			return t, true, nil
		}
	case Score:
		if x, isNum := v.(float64); isNum {
			if x < 0 || x > float64(len(q.Options)-1) || x != float64(int(x)) {
				return nil, false, fmt.Errorf("%s: invalid level %v", q.Name, x)
			}
			return []float64{x}, true, nil
		}
		if x, isStr := v.(string); isStr {
			if i := indexOf(q.Options, x); i >= 0 {
				return []float64{float64(i)}, true, nil
			}
			return nil, false, fmt.Errorf("%s: unknown level %q", q.Name, x)
		}
	}
	return nil, false, fmt.Errorf("%s: label %v (%T) incompatible with type %s", q.Name, v, v, q.Kind)
}

func indexOf(s []string, v string) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return -1
}
