// Package decision expose un modèle indecis comme fournisseur de décision de
// genai (llm.DecisionClient), à la place d'un service comme Jev :
//
//	import _ "github.com/bornholm/indecis/decision"
//
//	GENAI_DECISION_PROVIDER=indecis
//	GENAI_DECISION_INDECIS_MODEL=/chemin/vers/le/modèle
//
// Différence de fond avec Jev : Jev lit les instructions et les critères de
// chaque question et répond à n'importe laquelle. Un modèle indecis a appris
// un schéma fixe ; il répond aux questions de ce schéma, reconnues par leur
// identifiant, et ne lit pas leurs instructions. Une question inconnue du
// modèle est refusée avec la liste de celles qu'il connaît.
//
// L'état est jugé ainsi :
//   - une chaîne : le texte tel quel ;
//   - un objet {"context": …, "text": …} : une paire (prompt système,
//     message) pour un modèle construit avec indecis.WithPairs ;
//   - tout autre valeur : sa sérialisation JSON.
//
// Ce module est séparé d'indecis pour que la bibliothèque reste sans
// dépendance, et de genai parce qu'indecis est sous GPL.
package decision

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	"github.com/bornholm/indecis"
)

// Name est le nom du fournisseur dans la configuration de genai.
const Name provider.Name = "indecis"

// Options est lu depuis GENAI_DECISION_INDECIS_*.
type Options struct {
	// Model est le répertoire d'un modèle écrit par indecis.Model.Save.
	Model string `env:"MODEL"`
}

func init() {
	provider.RegisterDecision(Name,
		func() *Options { return &Options{} },
		func(ctx context.Context, opts *Options) (llm.DecisionClient, error) {
			return New(opts.Model)
		},
	)
}

// Client répond aux questions avec des modèles indecis locaux.
type Client struct {
	defaultDir string
	mu         sync.Mutex
	models     map[string]*indecis.Model
}

// New charge le modèle de dir, qui sert quand l'appel n'en désigne pas
// d'autre avec llm.WithDecisionModel.
func New(dir string) (*Client, error) {
	if dir == "" {
		return nil, fmt.Errorf("indecis : répertoire du modèle non configuré (GENAI_DECISION_INDECIS_MODEL)")
	}
	c := &Client{defaultDir: dir, models: map[string]*indecis.Model{}}
	if _, err := c.model(dir); err != nil {
		return nil, err
	}
	return c, nil
}

// FromModel enveloppe un modèle déjà chargé, désigné par name.
func FromModel(name string, m *indecis.Model) *Client {
	return &Client{defaultDir: name, models: map[string]*indecis.Model{name: m}}
}

func (c *Client) model(dir string) (*indecis.Model, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if m, ok := c.models[dir]; ok {
		return m, nil
	}
	m, err := indecis.Load(dir)
	if err != nil {
		return nil, fmt.Errorf("indecis : chargement de %s : %w", dir, err)
	}
	c.models[dir] = m
	return m, nil
}

// Decision implémente llm.DecisionClient.
func (c *Client) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	if err := questions.Validate(); err != nil {
		return nil, err
	}
	dir := c.defaultDir
	if opts := llm.NewDecisionOptions(funcs...); opts.Model != "" {
		dir = opts.Model
	}
	m, err := c.model(dir)
	if err != nil {
		return nil, err
	}
	in, err := stateInput(state, m.Paired())
	if err != nil {
		return nil, err
	}
	schema := map[string]indecis.Question{}
	for _, q := range m.Schema() {
		schema[q.Name] = q
	}
	for id, q := range questions {
		if err := compatible(id, q, schema); err != nil {
			return nil, err
		}
	}

	ds, err := m.DecideInputs(ctx, in)
	if err != nil {
		return nil, err
	}
	d := ds[0]
	answers := make(map[string]llm.Answer, len(questions))
	for id, q := range questions {
		answers[id] = toAnswer(q, schema[id], d[id])
	}
	tokens, _ := m.Tokens(in)
	return llm.NewDecisionResponse(filepath.Base(dir), answers, llm.NewDecisionUsage(int64(tokens), 0, int64(tokens))), nil
}

// stateInput convertit l'état en entrée du modèle.
func stateInput(state any, paired bool) (indecis.Input, error) {
	var in indecis.Input
	switch v := state.(type) {
	case string:
		in.Text = v
	case []byte:
		in.Text = string(v)
	case indecis.Input:
		in = v
	case *indecis.Input:
		in = *v
	case map[string]any:
		text, isText := v["text"].(string)
		ctx, isCtx := v["context"].(string)
		if isText && (isCtx || v["context"] == nil) && len(v) <= 2 {
			in = indecis.Input{Context: ctx, Text: text}
			break
		}
		b, err := json.Marshal(v)
		if err != nil {
			return in, fmt.Errorf("indecis : état non sérialisable : %w", err)
		}
		in.Text = string(b)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return in, fmt.Errorf("indecis : état non sérialisable : %w", err)
		}
		in.Text = string(b)
	}
	if in.Context != "" && !paired {
		return in, llm.NewValidationError("state", "ce modèle indecis ne lit pas de contexte : passer le texte seul")
	}
	return in, nil
}

// compatible vérifie qu'une question de la requête correspond à une question
// apprise par le modèle.
func compatible(id string, q llm.Question, schema map[string]indecis.Question) error {
	sq, ok := schema[id]
	if !ok {
		names := make([]string, 0, len(schema))
		for n, s := range schema {
			names = append(names, fmt.Sprintf("%s (%s)", n, s.Kind))
		}
		sort.Strings(names)
		return llm.NewValidationError("questions."+id, fmt.Sprintf(
			"question inconnue du modèle : un modèle indecis ne répond qu'aux questions apprises, identifiées par leur nom : %s",
			strings.Join(names, ", ")))
	}
	if string(q.QuestionType()) != string(sq.Kind) {
		return llm.NewValidationError("questions."+id, fmt.Sprintf("le modèle répond à %q par un %s, pas un %s", id, sq.Kind, q.QuestionType()))
	}
	switch v := q.(type) {
	case llm.ChoiceQuestion:
		known := map[string]bool{}
		for _, o := range sq.Options {
			known[o] = true
		}
		for o := range v.Criteria {
			if !known[o] {
				return llm.NewValidationError("questions."+id+".criteria", fmt.Sprintf("option %q inconnue du modèle (options apprises : %s)", o, strings.Join(sq.Options, ", ")))
			}
		}
	case llm.ScoreQuestion:
		if len(v.Criteria) != len(sq.Options) {
			return llm.NewValidationError("questions."+id+".criteria", fmt.Sprintf("le modèle note sur %d niveaux (%s), pas %d", len(sq.Options), strings.Join(sq.Options, " < "), len(v.Criteria)))
		}
	}
	return nil
}

func toAnswer(q llm.Question, sq indecis.Question, a indecis.Answer) llm.Answer {
	switch v := q.(type) {
	case llm.ChoiceQuestion:
		// Distribution renormalisée sur les options demandées.
		probs := map[string]float64{}
		var sum float64
		for o := range v.Criteria {
			probs[o] = a.Probs[o]
			sum += a.Probs[o]
		}
		best, conf := "", -1.0
		for o, p := range probs {
			if sum > 0 {
				p /= sum
			} else {
				p = 1 / float64(len(probs))
			}
			probs[o] = p
			if p > conf || (p == conf && o < best) {
				best, conf = o, p
			}
		}
		return llm.NewChoiceAnswer(best, probs, conf)
	case llm.ScoreQuestion:
		// Niveaux indexés par leur position, comme chez TypeSafe ; l'ordre
		// est celui du schéma appris, la légende celle de la requête.
		probs := map[string]float64{}
		legend := map[string]string{}
		conf := 0.0
		for i, c := range v.Criteria {
			p := a.Probs[sq.Options[i]]
			probs[strconv.Itoa(i)] = p
			legend[strconv.Itoa(i)] = describe(c)
			conf = math.Max(conf, p)
		}
		return llm.NewScoreAnswer(a.Score, legend, probs, conf)
	default:
		return llm.NewNoulAnswer(a.P)
	}
}

func describe(c any) string {
	if s, ok := c.(string); ok {
		return s
	}
	b, err := json.Marshal(c)
	if err != nil {
		return fmt.Sprint(c)
	}
	return string(b)
}

var _ llm.DecisionClient = (*Client)(nil)
