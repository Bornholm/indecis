// Package vision takes decisions on images with a SigLIP model, without
// training: the options of a question are described in text, and each is
// scored against the image by the model, as indecis does for texts in open
// mode (see indecis.OpenQuestion).
//
//	m, _ := vision.Load(dir, vision.WithInt8())
//	a, _ := m.ChooseNearest(ctx, []indecis.Candidate{
//	    {Name: "invoice", Description: "a scanned invoice"},
//	    {Name: "photo", Description: "a photo of people"},
//	}, img)
//
// The supported checkpoints are transformers' "siglip" models, such as
// google/siglip2-base-patch32-256. SigLIP scores each (image, text) pair
// with its own sigmoid: Match is the probability that a text describes an
// image, on its own, without comparing it to other texts.
package vision

import (
	"context"
	"fmt"
	"image"
	"math"
	"path/filepath"
	"sync"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/internal/siglip"
	"github.com/bornholm/indecis/tokenizer"
)

// Model is a SigLIP model ready to decide.
type Model struct {
	m   *siglip.Model
	tok *tokenizer.Tokenizer

	mu    sync.Mutex
	cache map[string][]float32 // text embeddings: options recur from call to call
}

// maxCache bounds the text embedding cache; beyond, it starts over.
const maxCache = 4096

type config struct{ int8 bool }

// Option configures Load.
type Option func(*config)

// WithInt8 computes most layer products in int8 when the processor has
// AVX-VNNI. On Imagenette, zero-shot accuracy is unchanged.
func WithInt8() Option { return func(c *config) { c.int8 = true } }

// Load reads a SigLIP model directory: config.json, model.safetensors and
// tokenizer.json.
func Load(dir string, opts ...Option) (*Model, error) {
	var c config
	for _, o := range opts {
		o(&c)
	}
	m, err := siglip.Load(dir, c.int8)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadShared(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	return &Model{m: m, tok: tok, cache: map[string][]float32{}}, nil
}

// EmbedImage returns the normalized embedding of an image.
func (m *Model) EmbedImage(img image.Image) ([]float32, error) {
	px, err := m.m.Cfg.Preprocess(img)
	if err != nil {
		return nil, err
	}
	e, err := m.m.Vision.Embed(px)
	if err != nil {
		return nil, err
	}
	return normalize(e), nil
}

// EmbedText returns the normalized embedding of a text, in the image
// embeddings' space. Texts longer than the tower (64 tokens) are truncated.
func (m *Model) EmbedText(text string) ([]float32, error) {
	m.mu.Lock()
	e, ok := m.cache[text]
	m.mu.Unlock()
	if ok {
		return e, nil
	}
	ids := m.m.Text.Pad(m.tok.Encode(text), m.tok.EosID(), m.tok.PadID())
	e, err := m.m.Text.Embed(ids)
	if err != nil {
		return nil, err
	}
	e = normalize(e)
	m.mu.Lock()
	if len(m.cache) >= maxCache {
		m.cache = map[string][]float32{}
	}
	m.cache[text] = e
	m.mu.Unlock()
	return e, nil
}

// logit is the model's score for an (image, text) pair of normalized
// embeddings; its sigmoid is the probability that the text describes the
// image.
func (m *Model) logit(img, text []float32) float64 {
	return float64(m.m.Logit(dot(img, text)))
}

// Match returns the probability that text describes img.
func (m *Model) Match(img image.Image, text string) (float64, error) {
	ie, err := m.EmbedImage(img)
	if err != nil {
		return 0, err
	}
	te, err := m.EmbedText(text)
	if err != nil {
		return 0, err
	}
	return sigmoid(m.logit(ie, te)), nil
}

// optionText is what the text tower reads for an option: its description,
// written as a caption ("a photo of a cat"), or its name.
func optionText(c indecis.Candidate) string {
	if c.Description != "" {
		return c.Description
	}
	return c.Name
}

// ChooseNearest picks the option that best describes img: the softmax of
// the model's logits over the options gives Probs.
func (m *Model) ChooseNearest(ctx context.Context, cands []indecis.Candidate, img image.Image) (indecis.Answer, error) {
	ie, err := m.EmbedImage(img)
	if err != nil {
		return indecis.Answer{}, err
	}
	return m.choose(ie, cands)
}

func (m *Model) choose(ie []float32, cands []indecis.Candidate) (indecis.Answer, error) {
	if len(cands) < 2 {
		return indecis.Answer{}, fmt.Errorf("vision: at least two options")
	}
	logits := make([]float64, len(cands))
	cos := make([]float64, len(cands))
	for i, c := range cands {
		te, err := m.EmbedText(optionText(c))
		if err != nil {
			return indecis.Answer{}, err
		}
		cos[i], logits[i] = float64(dot(ie, te)), m.logit(ie, te)
	}
	p := softmax(logits)
	best := 0
	for i := range p {
		if p[i] > p[best] {
			best = i
		}
	}
	a := indecis.Answer{Kind: indecis.Choice, Probs: make(map[string]float64, len(p))}
	for i, c := range cands {
		a.Probs[c.Name] = p[i]
	}
	a.Choice, a.Confidence, a.Score = cands[best].Name, p[best], cos[best]
	if len(p) > 1 {
		var others float64
		for i, v := range p {
			if i != best {
				others += v
			}
		}
		a.Margin = p[best] - others/float64(len(p)-1)
	}
	return a, nil
}

// DecideOpen answers open questions on each image:
//   - Choice: the option that best describes the image, as ChooseNearest;
//   - Score: the levels as options, Score being the expected level;
//   - Noul: with the default options of indecis.OpenNoul, P is Match of
//     the instructions (SigLIP scores a text on its own); with two described
//     criteria, the softmax of their logits.
//
// Questions' Instructions are not read for Choice and Score: the text
// tower reads captions, so describe each option as one.
func (m *Model) DecideOpen(ctx context.Context, questions []indecis.OpenQuestion, images ...image.Image) ([]indecis.Decision, error) {
	out := make([]indecis.Decision, len(images))
	for i, img := range images {
		ie, err := m.EmbedImage(img)
		if err != nil {
			return nil, err
		}
		out[i] = indecis.Decision{}
		for _, q := range questions {
			a, err := m.decide(ie, q)
			if err != nil {
				return nil, fmt.Errorf("vision: %s: %w", q.Name, err)
			}
			a.Question = q.Name
			out[i][q.Name] = a
		}
	}
	return out, nil
}

func (m *Model) decide(ie []float32, q indecis.OpenQuestion) (indecis.Answer, error) {
	switch q.Kind {
	case indecis.Noul:
		if len(q.Options) != 2 {
			return indecis.Answer{}, fmt.Errorf("an open noul question describes two criteria (true, false)")
		}
		var p float64
		if q.Options[1].Description == indecis.NoulAnchor {
			te, err := m.EmbedText(optionText(q.Options[0]))
			if err != nil {
				return indecis.Answer{}, err
			}
			p = sigmoid(m.logit(ie, te))
		} else {
			a, err := m.choose(ie, q.Options)
			if err != nil {
				return indecis.Answer{}, err
			}
			p = a.Probs[q.Options[0].Name]
		}
		return indecis.Answer{Kind: indecis.Noul, P: p, Confidence: math.Max(p, 1-p)}, nil
	case indecis.Choice, indecis.Score:
		a, err := m.choose(ie, q.Options)
		if err != nil {
			return a, err
		}
		a.Kind = q.Kind
		if q.Kind == indecis.Score {
			a.Score = 0
			for k, o := range q.Options {
				a.Score += float64(k) * a.Probs[o.Name]
			}
		}
		return a, nil
	}
	return indecis.Answer{}, fmt.Errorf("unknown type %q", q.Kind)
}

func normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func softmax(z []float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range z {
		mx = math.Max(mx, v)
	}
	p := make([]float64, len(z))
	var s float64
	for i, v := range z {
		p[i] = math.Exp(v - mx)
		s += p[i]
	}
	for i := range p {
		p[i] /= s
	}
	return p
}

// IsModel reports whether dir holds a model this package reads (its
// config.json declares the "siglip" model type).
func IsModel(dir string) bool {
	_, err := siglip.ReadConfig(dir)
	return err == nil
}
