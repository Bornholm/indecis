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
// dependency.
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
	"github.com/bornholm/indecis/vision"
)

// Name is the provider name in genai's configuration.
const Name provider.Name = "indecis"

// Options is read from GENAI_DECISION_INDECIS_*.
type Options struct {
	// Model is the directory of a model written by indecis.Model.Save.
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

// Client answers questions with local indecis models.
type Client struct {
	defaultDir string
	mu         sync.Mutex
	models     map[string]*indecis.Model
	visions    map[string]*vision.Model // image models (SigLIP), see image.go
	// MaxImagePixels bounds the images an image model accepts (0:
	// DefaultMaxImagePixels).
	MaxImagePixels int
}

// New loads the model from dir, which serves when the call does not
// designate another one with llm.WithDecisionModel.
func New(dir string) (*Client, error) {
	if dir == "" {
		return nil, fmt.Errorf("indecis: model directory not configured (GENAI_DECISION_INDECIS_MODEL)")
	}
	c := &Client{defaultDir: dir, models: map[string]*indecis.Model{}}
	if _, err := c.model(dir); err != nil {
		return nil, err
	}
	return c, nil
}

// FromModel wraps an already loaded model, designated by name.
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
		return nil, fmt.Errorf("indecis: loading %s: %w", dir, err)
	}
	c.models[dir] = m
	return m, nil
}

// Decision implements llm.DecisionClient.
func (c *Client) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	if err := questions.Validate(); err != nil {
		return nil, err
	}
	dir := c.defaultDir
	if opts := llm.NewDecisionOptions(funcs...); opts.Model != "" {
		dir = opts.Model
	}
	if v, err := c.visionModel(dir); err != nil {
		return nil, err
	} else if v != nil {
		return decideImage(ctx, filepath.Base(dir), v, state, questions, c.MaxImagePixels)
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
	// A learned question (same name as in the model's schema) goes
	// through its calibrated head. Any other question is open: its
	// criteria are compared to the state by embeddings (DecideOpen), as
	// with Jev.
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

// stateInput converts the state into a model input.
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
			return in, fmt.Errorf("indecis: unserializable state: %w", err)
		}
		in.Text = string(b)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return in, fmt.Errorf("indecis: unserializable state: %w", err)
		}
		in.Text = string(b)
	}
	if in.Context != "" && !paired {
		return in, llm.NewValidationError("state", "this indecis model does not read a context: pass the text alone")
	}
	return in, nil
}

// compatible checks that a question of the request matches a question
// learned by the model.
func compatible(id string, q llm.Question, schema map[string]indecis.Question) error {
	sq := schema[id]
	if string(q.QuestionType()) != string(sq.Kind) {
		return llm.NewValidationError("questions."+id, fmt.Sprintf("the model answers %q with a %s, not a %s", id, sq.Kind, q.QuestionType()))
	}
	switch v := q.(type) {
	case llm.ChoiceQuestion:
		known := map[string]bool{}
		for _, o := range sq.Options {
			known[o] = true
		}
		for o := range v.Criteria {
			if !known[o] {
				return llm.NewValidationError("questions."+id+".criteria", fmt.Sprintf("option %q unknown to the model (learned options: %s)", o, strings.Join(sq.Options, ", ")))
			}
		}
	case llm.ScoreQuestion:
		if len(v.Criteria) != len(sq.Options) {
			return llm.NewValidationError("questions."+id+".criteria", fmt.Sprintf("the model scores on %d levels (%s), not %d", len(sq.Options), strings.Join(sq.Options, " < "), len(v.Criteria)))
		}
	}
	return nil
}

func toAnswer(q llm.Question, sq indecis.Question, a indecis.Answer) llm.Answer {
	switch v := q.(type) {
	case llm.ChoiceQuestion:
		// Distribution renormalized over the requested options.
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
		// Levels indexed by their position, as with TypeSafe; the order
		// is that of the learned schema, the legend that of the request.
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

// toOpen translates a genai question into an open question: the criteria
// become described options. A yes/no question with no criteria opposes its
// instructions to a fixed anchor (see indecis.OpenNoul).
//
// A criterion's description can be an object {"description": ...,
// "examples": [...]}, a form the TypeSafe API allows (a description is a
// string, an object or an array): the examples then situate the option
// (see indecis.Candidate.Examples). This is an indecis convention; another
// provider reads the object as a description.
func toOpen(id string, q llm.Question) indecis.OpenQuestion {
	switch v := q.(type) {
	case llm.NoulQuestion:
		oq := indecis.OpenNoul(id, describe(v.Instructions))
		yes, no := oq.Options[0], oq.Options[1]
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

// criterion reads a criterion's description: a string, nil, or an
// object whose "examples" (list of texts) gives examples and
// "description" the description; the other fields of an object stay in
// the description.
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

// texts accepts a list of texts, whether it comes from Go ([]string) or
// from decoded JSON ([]any).
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
