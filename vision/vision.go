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
	m    *siglip.Model
	head *Head // learned questions, nil for a bare encoder

	// The tokenizer and the text tower load at the first text: a model
	// whose questions are all learned never needs them.
	tokPath string
	tokOnce sync.Once
	tok     *tokenizer.Tokenizer
	tokErr  error

	mu       sync.Mutex
	cache    map[string][]float32 // text embeddings: options recur from call to call
	maxCache int

	pixels sync.Pool // *[]float32: normalized pixels, reused from image to image
}

// defaultCache bounds the text embedding cache; beyond, it starts over.
const defaultCache = 4096

type config struct {
	int8    bool
	threads int
	cache   int
}

// Option configures Load.
type Option func(*config)

// WithInt8 computes most layer products in int8 when the processor has
// AVX-VNNI. On Imagenette, zero-shot accuracy is unchanged.
func WithInt8() Option { return func(c *config) { c.int8 = true } }

// WithThreads spreads one image over n cores (default 1; 0: all). The
// other cores stay free for simultaneous requests; on a hybrid processor,
// use at most the number of performance cores.
func WithThreads(n int) Option { return func(c *config) { c.threads = n } }

// WithEmbedCache keeps the embeddings of up to n texts (the options of the
// questions), 4,096 by default; 0 disables the cache.
func WithEmbedCache(n int) Option { return func(c *config) { c.cache = max(n, 0) } }

// Load reads a SigLIP model directory (config.json, model.safetensors and
// tokenizer.json), or a trained model directory (see SaveHead), which
// loads its encoder and its head.
func Load(dir string, opts ...Option) (*Model, error) {
	c := config{threads: 1, cache: defaultCache}
	for _, o := range opts {
		o(&c)
	}
	man, err := readManifest(dir)
	if err != nil {
		return nil, err
	}
	var head *Head
	if man != nil {
		if head, err = loadHead(dir, man); err != nil {
			return nil, err
		}
		dir = man.Backbone
	}
	m, err := siglip.Load(dir, c.int8)
	if err != nil {
		return nil, err
	}
	m.SetThreads(c.threads)

	if head != nil && (head.T != m.Cfg.Patches() || head.H != m.Cfg.Hidden) {
		return nil, fmt.Errorf("vision: the head expects %d×%d patch features, the encoder gives %d×%d", head.T, head.H, m.Cfg.Patches(), m.Cfg.Hidden)
	}
	return &Model{m: m, tokPath: filepath.Join(dir, "tokenizer.json"), head: head, cache: map[string][]float32{}, maxCache: c.cache}, nil
}

// Schema returns the learned questions, nil for a bare encoder.
func (m *Model) Schema() indecis.Schema {
	if m.head == nil {
		return nil
	}
	return m.head.Schema
}

// Patches returns the features of each patch of img, [patches, hidden]:
// what a Head reads.
func (m *Model) Patches(img image.Image) ([]float32, error) {
	return m.PatchesAt(img, 0)
}

// PatchesAt returns the patch features after layer layers of the encoder
// (0: the final ones), computing no further.
func (m *Model) PatchesAt(img image.Image, layer int) ([]float32, error) {
	px, err := m.preprocess(img)
	if err != nil {
		return nil, err
	}
	p, _, err := m.m.Vision.Forward(px, layer, false)
	m.release(px)
	return p, err
}

// PatchShape returns the number of patches and their width.
func (m *Model) PatchShape() (patches, hidden int) { return m.m.Cfg.Patches(), m.m.Cfg.Hidden }

// Decide answers for img, with one pass of the encoder, the learned
// questions named in learned (the model's head) and open questions (see
// DecideOpen).
func (m *Model) Decide(ctx context.Context, img image.Image, learned []string, open []indecis.OpenQuestion) (indecis.Decision, error) {
	if len(learned) > 0 && m.head == nil {
		return nil, fmt.Errorf("vision: this model has no learned questions")
	}
	px, err := m.preprocess(img)
	if err != nil {
		return nil, err
	}
	layer := 0
	if m.head != nil {
		layer = m.head.Layer
	}
	// Without open questions, the encoder stops at the head's layer.
	patches, pooled, err := m.m.Vision.Forward(px, layer, len(open) > 0)
	m.release(px)
	if err != nil {
		return nil, err
	}
	d := indecis.Decision{}
	if len(learned) > 0 {
		hd, err := m.head.Decide(patches)
		if err != nil {
			return nil, err
		}
		for _, name := range learned {
			a, ok := hd[name]
			if !ok {
				return nil, fmt.Errorf("vision: %q is not a learned question", name)
			}
			d[name] = a
		}
	}
	if len(open) == 0 {
		return d, nil
	}
	ie := normalize(pooled)
	for _, q := range open {
		a, err := m.decide(ie, q)
		if err != nil {
			return nil, fmt.Errorf("vision: %s: %w", q.Name, err)
		}
		a.Question = q.Name
		d[q.Name] = a
	}
	return d, nil
}

// EmbedImage returns the normalized embedding of an image.
func (m *Model) EmbedImage(img image.Image) ([]float32, error) {
	px, err := m.preprocess(img)
	if err != nil {
		return nil, err
	}
	e, err := m.m.Vision.Embed(px)
	m.release(px)
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
	m.tokOnce.Do(func() { m.tok, m.tokErr = tokenizer.LoadShared(m.tokPath) })
	if m.tokErr != nil {
		return nil, m.tokErr
	}
	tt, err := m.m.Text()
	if err != nil {
		return nil, err
	}
	ids := tt.Pad(m.tok.Encode(text), m.tok.EosID(), m.tok.PadID())
	e, err = tt.Embed(ids)
	if err != nil {
		return nil, err
	}
	e = normalize(e)
	m.mu.Lock()
	if m.maxCache == 0 {
		m.mu.Unlock()
		return e, nil
	}
	if len(m.cache) >= m.maxCache {
		m.cache = map[string][]float32{}
	}
	m.cache[text] = e
	m.mu.Unlock()
	return e, nil
}

// Logit is the model's score for an (image, text) pair of normalized
// embeddings (EmbedImage, EmbedText); its sigmoid is the probability that
// the text describes the image.
func (m *Model) Logit(img, text []float32) float64 {
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
	return sigmoid(m.Logit(ie, te)), nil
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
		cos[i], logits[i] = float64(dot(ie, te)), m.Logit(ie, te)
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
//   - Noul: P is the softmax of the logits of the two criteria, by default
//     the instructions against indecis.NoulAnchor (see indecis.OpenNoul);
//     describing the negation ("a photo of an object") does better.
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
		// SigLIP's own sigmoid (Match) ranks images well but is set for
		// precise captions: on Imagenette, "a photo of an animal" stayed
		// under 0.5 for most animals. Opposing the two criteria, by default
		// the instructions and indecis.NoulAnchor, answered 86 to 97% right
		// at 0.5; a described negation ("a photo of an object"), 92 to 98%.
		a, err := m.choose(ie, q.Options)
		if err != nil {
			return indecis.Answer{}, err
		}
		p := a.Probs[q.Options[0].Name]
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

// IsModel reports whether dir holds a model this package reads: a SigLIP
// encoder (config.json of the "siglip" model type) or a trained model
// (vision.json).
func IsModel(dir string) bool {
	if m, err := readManifest(dir); err == nil && m != nil {
		return true
	}
	_, err := siglip.ReadConfig(dir)
	return err == nil
}

// preprocess returns the normalized pixels of img in a buffer from the
// pool; release gives it back once the encoder has read it.
func (m *Model) preprocess(img image.Image) ([]float32, error) {
	var dst []float32
	if p, ok := m.pixels.Get().(*[]float32); ok {
		dst = *p
	}
	return m.m.Cfg.PreprocessInto(dst, img)
}

func (m *Model) release(px []float32) {
	if px != nil {
		m.pixels.Put(&px)
	}
}
