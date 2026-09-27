package indecis

import (
	"context"
	"fmt"
)

// OpenQuestion est une question posée à l'inférence, sans entraînement
// dédié : ses réponses possibles sont décrites en texte et comparées au
// texte à juger par plongements (voir ChooseIn). C'est la forme des
// questions de Jev et de CLM.
type OpenQuestion struct {
	Name string
	Kind Kind
	// Instructions, si fournies, précèdent chaque option dans ce que lit
	// le modèle (« Urgence du courriel » avant « haute »).
	Instructions string
	// Options : pour Choice, les options ; pour Score, les niveaux du plus
	// bas au plus haut ; pour Noul, exactement deux critères, celui du
	// « vrai » puis celui du « faux ».
	Options []Candidate
}

func (q OpenQuestion) validate() error {
	switch q.Kind {
	case Noul:
		if len(q.Options) != 2 {
			return fmt.Errorf("indecis: %s : une question noul ouverte décrit deux critères (vrai, faux)", q.Name)
		}
	case Choice, Score:
		if len(q.Options) < 2 {
			return fmt.Errorf("indecis: %s : au moins deux options", q.Name)
		}
	default:
		return fmt.Errorf("indecis: %s : type %q inconnu", q.Name, q.Kind)
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

// DecideOpen répond à des questions ouvertes pour chaque texte. Les
// probabilités viennent du softmax des cosinus (voir ChooseIn) : leur
// calibration dépend du modèle, et un seuil se règle sur des exemples.
// WithEmbedCache évite de réencoder les options d'un appel à l'autre.
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
