package indecis

import (
	"context"
	"fmt"
)

// OpenQuestion is a question asked at inference time, without dedicated
// training: its possible answers are described in text and compared to the
// text being judged by embeddings (see ChooseIn). This is the form of the
// Jev and CLM questions.
type OpenQuestion struct {
	Name string
	Kind Kind
	// Instructions, if given, precede each option in what the model reads
	// ("Email urgency" before "high").
	Instructions string
	// Options: for Choice, the options; for Score, the levels from lowest
	// to highest; for Noul, exactly two criteria, the "true" one then the
	// "false" one.
	Options []Candidate
}

func (q OpenQuestion) validate() error {
	switch q.Kind {
	case Noul:
		if len(q.Options) != 2 {
			return fmt.Errorf("indecis: %s: an open noul question describes two criteria (true, false)", q.Name)
		}
	case Choice, Score:
		if len(q.Options) < 2 {
			return fmt.Errorf("indecis: %s: at least two options", q.Name)
		}
	default:
		return fmt.Errorf("indecis: %s: unknown type %q", q.Name, q.Kind)
	}
	return nil
}

func (q OpenQuestion) candidates() []Candidate {
	out := make([]Candidate, len(q.Options))
	for i, o := range q.Options {
		if q.Instructions != "" {
			o.Description = joinNonEmpty(q.Instructions, o.Description)
		}
		out[i] = o
	}
	return out
}

func joinNonEmpty(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	}
	return a + "\n" + b
}

// DecideOpen answers open questions for each text. The probabilities come
// from the softmax of the cosines (see ChooseIn): their calibration
// depends on the model, and a threshold is tuned on examples.
// WithEmbedCache avoids re-encoding the options from one call to the next.
func (m *Model) DecideOpen(ctx context.Context, questions []OpenQuestion, texts ...string) ([]Decision, error) {
	out := make([]Decision, len(texts))
	for i := range out {
		out[i] = Decision{}
	}
	for _, q := range questions {
		if err := q.validate(); err != nil {
			return nil, err
		}
		set, err := m.PrepareCandidates(ctx, q.candidates())
		if err != nil {
			return nil, fmt.Errorf("indecis: %s : %w", q.Name, err)
		}
		answers, err := m.ChooseIn(ctx, set, texts...)
		if err != nil {
			return nil, err
		}
		for i, a := range answers {
			a.Question, a.Kind = q.Name, q.Kind
			switch q.Kind {
			case Noul:
				a.P = a.Probs[q.Options[0].Name]
				a.Confidence = max(a.P, 1-a.P)
				a.Choice, a.Probs, a.Margin = "", nil, 0
			case Score:
				for k, o := range q.Options {
					a.Score += float64(k) * a.Probs[o.Name]
				}
			}
			out[i][q.Name] = a
		}
	}
	return out, nil
}
