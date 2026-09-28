// Package decision exposes indecis models as a genai decision provider
// (llm.DecisionClient), in place of a service such as Jev, and as an HTTP
// server compatible with the TypeSafe and OpenRouter decision API
// ([Server]).
//
//	import _ "github.com/bornholm/indecis/decision"
//
//	GENAI_DECISION_PROVIDER=indecis
//	GENAI_DECISION_INDECIS_MODEL=/path/to/the/model
//
// A question the model learned, recognized by its identifier, is answered by
// its trained head; its instructions are not read. Any other question is
// asked in open mode: its criteria are compared with the state through
// embeddings (see indecis.Model.DecideOpen).
//
// The state is judged as follows:
//   - a string: the text as is;
//   - an object {"context": …, "text": …}: a (system prompt, message) pair
//     for a model built with indecis.WithPairs;
//   - any other value: its JSON serialization.
//
// This module is separate from indecis so that the library keeps no
// dependency, and separate from genai because indecis is under the GPL.
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
	// Une question apprise (même nom que dans le schéma du modèle) passe
	// par sa tête, calibrée. Toute autre question est ouverte : ses
	// critères sont comparés à l'état par plongements (DecideOpen), comme
	// chez Jev.
	var ids []string
	var open []indecis.OpenQuestion
	learned := false
	for id, q := range questions {
		if _, ok := schema[id]; !ok {
			ids = append(ids, id)
			open = append(open, toOpen(id, q))
			continue
		}
		if err := compatible(id, q, schema); err != nil {
			return nil, err
		}
		learned = true
	}
	answers := make(map[string]llm.Answer, len(questions))
	if learned {
		ds, err := m.DecideInputs(ctx, in)
		if err != nil {
			return nil, err
		}
		for id, q := range questions {
			if _, ok := schema[id]; ok {
				answers[id] = toAnswer(q, schema[id], ds[0][id])
			}
		}
	}
	if len(open) > 0 {
		text := in.Text
		if in.Context != "" {
			text = in.Context + "\n\n" + text
		}
		ds, err := m.DecideOpen(ctx, open, text)
		if err != nil {
			return nil, err
		}
		for k, id := range ids {
			answers[id] = toOpenAnswer(questions[id], ds[0][open[k].Name])
		}
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
	sq := schema[id]
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

// toOpen traduit une question genai en question ouverte : les critères
// deviennent des options décrites. Une question oui/non sans critères
// oppose l'affirmation et la négation de ses instructions.
//
// Une description de critère peut être un objet {"description": …,
// "examples": […]}, forme que l'API TypeSafe admet (une description est
// une chaîne, un objet ou un tableau) : les exemples situent alors
// l'option (voir indecis.Candidate.Examples). C'est une convention
// d'indecis ; un autre fournisseur lit l'objet comme une description.
func toOpen(id string, q llm.Question) indecis.OpenQuestion {
	switch v := q.(type) {
	case llm.NoulQuestion:
		instr := describe(v.Instructions)
		yes := indecis.Candidate{Name: "true", Description: "Oui : " + instr}
		no := indecis.Candidate{Name: "false", Description: "Non, pas du tout : " + instr}
		if v.True != nil {
			yes = criterion("true", v.True)
		}
		if v.False != nil {
			no = criterion("false", v.False)
		}
		return indecis.OpenQuestion{Name: id, Kind: indecis.Noul, Options: []indecis.Candidate{yes, no}}
	case llm.ChoiceQuestion:
		names := make([]string, 0, len(v.Criteria))
		for o := range v.Criteria {
			names = append(names, o)
		}
		sort.Strings(names)
		oq := indecis.OpenQuestion{Name: id, Kind: indecis.Choice, Instructions: describe(v.Instructions)}
		for _, n := range names {
			oq.Options = append(oq.Options, criterion(n, v.Criteria[n]))
		}
		return oq
	case llm.ScoreQuestion:
		oq := indecis.OpenQuestion{Name: id, Kind: indecis.Score, Instructions: describe(v.Instructions)}
		for i, c := range v.Criteria {
			oq.Options = append(oq.Options, criterion(strconv.Itoa(i), c))
		}
		return oq
	}
	return indecis.OpenQuestion{Name: id}
}

// criterion lit la description d'un critère : une chaîne, nil, ou un
// objet dont « examples » (liste de textes) donne des exemples et
// « description » la description ; les autres champs d'un objet restent
// dans la description.
func criterion(name string, v any) indecis.Candidate {
	c := indecis.Candidate{Name: name}
	obj, ok := v.(map[string]any)
	if !ok {
		if v != nil {
			c.Description = describe(v)
		}
		return c
	}
	rest := map[string]any{}
	for k, x := range obj {
		switch k {
		case "examples":
			c.Examples = append(c.Examples, texts(x)...)
		case "description":
			c.Description = describe(x)
		default:
			rest[k] = x
		}
	}
	if len(rest) > 0 {
		c.Description = strings.TrimSpace(c.Description + "\n" + describe(rest))
	}
	return c
}

// texts accepte une liste de textes, qu'elle vienne de Go ([]string) ou
// d'un JSON décodé ([]any).
func texts(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		var out []string
		for _, e := range x {
			if s, ok := e.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	case string:
		return []string{x}
	}
	return nil
}

func toOpenAnswer(q llm.Question, a indecis.Answer) llm.Answer {
	switch v := q.(type) {
	case llm.ChoiceQuestion:
		return llm.NewChoiceAnswer(a.Choice, a.Probs, a.Confidence)
	case llm.ScoreQuestion:
		legend := map[string]string{}
		for i, c := range v.Criteria {
			legend[strconv.Itoa(i)] = describe(c)
		}
		return llm.NewScoreAnswer(a.Score, legend, a.Probs, a.Confidence)
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
