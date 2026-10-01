// Package siglip implements the SigLIP image and text encoders in pure Go,
// for inference: a Vision Transformer that turns an image into an
// embedding, and a text transformer whose embeddings live in the same
// space. Comparing the two decides whether a description fits an image,
// with no training (zero-shot).
//
// The checkpoints are those of transformers ("siglip" model type, as
// google/siglip2-base-patch32-256): pre-norm transformer layers with
// biases, a tanh GELU MLP, learned position embeddings, attention pooling
// for the image and the last token for the text. Parity with transformers
// is tested on fixtures from tools/oracle/siglip_fixtures.py.
package siglip

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/safetensors"
)

// Config holds the dimensions of the two towers. Zero fields of
// config.json take the defaults of transformers' SiglipConfig.
type Config struct {
	Hidden, Layers, Heads, MLP int
	Eps                        float64
	ImageSize, PatchSize       int
	TextLen, Vocab             int
}

// Patches is the number of image patches, the vision sequence length.
func (c Config) Patches() int { g := c.ImageSize / c.PatchSize; return g * g }

func (c Config) headDim() int { return c.Hidden / c.Heads }

type configJSON struct {
	ModelType    string `json:"model_type"`
	VisionConfig struct {
		Hidden    int     `json:"hidden_size"`
		Layers    int     `json:"num_hidden_layers"`
		Heads     int     `json:"num_attention_heads"`
		MLP       int     `json:"intermediate_size"`
		Eps       float64 `json:"layer_norm_eps"`
		ImageSize int     `json:"image_size"`
		PatchSize int     `json:"patch_size"`
		Act       string  `json:"hidden_act"`
	} `json:"vision_config"`
	TextConfig struct {
		Hidden int     `json:"hidden_size"`
		Layers int     `json:"num_hidden_layers"`
		Heads  int     `json:"num_attention_heads"`
		Eps    float64 `json:"layer_norm_eps"`
		MLP    int     `json:"intermediate_size"`
		MaxPos int     `json:"max_position_embeddings"`
		Vocab  int     `json:"vocab_size"`
		Act    string  `json:"hidden_act"`
	} `json:"text_config"`
}

// ReadConfig reads config.json from a model directory.
func ReadConfig(dir string) (Config, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return Config{}, err
	}
	var j configJSON
	if err := json.Unmarshal(b, &j); err != nil {
		return Config{}, fmt.Errorf("siglip: config.json: %w", err)
	}
	if j.ModelType != "siglip" {
		return Config{}, fmt.Errorf("siglip: model type %q not supported (expected siglip)", j.ModelType)
	}
	v, t := j.VisionConfig, j.TextConfig
	for _, act := range []string{v.Act, t.Act} {
		if act != "" && act != "gelu_pytorch_tanh" {
			return Config{}, fmt.Errorf("siglip: activation %q not supported", act)
		}
	}
	or := func(x, def int) int {
		if x == 0 {
			return def
		}
		return x
	}
	c := Config{
		Hidden: or(v.Hidden, 768), Layers: or(v.Layers, 12), Heads: or(v.Heads, 12), MLP: or(v.MLP, 3072),
		Eps: v.Eps, ImageSize: or(v.ImageSize, 224), PatchSize: or(v.PatchSize, 16),
		TextLen: or(t.MaxPos, 64), Vocab: or(t.Vocab, 32000),
	}
	if c.Eps == 0 {
		c.Eps = 1e-6
	}
	// The text tower shares the vision dimensions in every released
	// checkpoint; a model where they differ is refused rather than misread.
	if (t.Hidden != 0 && t.Hidden != c.Hidden) || (t.Layers != 0 && t.Layers != c.Layers) || (t.Heads != 0 && t.Heads != c.Heads) ||
		(t.Eps != 0 && t.Eps != c.Eps) || (t.MLP != 0 && t.MLP != c.MLP) {
		return Config{}, fmt.Errorf("siglip: text and vision towers of different sizes are not supported")
	}
	if c.Hidden%c.Heads != 0 || c.ImageSize%c.PatchSize != 0 {
		return Config{}, fmt.Errorf("siglip: inconsistent dimensions")
	}
	return c, nil
}

// Model is a SigLIP checkpoint: both towers and the logit scale and bias.
// The text tower is loaded on first use (Text): a model whose questions
// are all learned by a head never needs it.
type Model struct {
	Cfg        Config
	Vision     *Vision
	LogitScale float32 // log of the temperature
	LogitBias  float32
	file       *safetensors.File
	quantized  bool
	workers    int

	textMu sync.Mutex // guards text; held while it loads
	text   *Text
}

// Text returns the text tower, loading it the first time (a few seconds,
// about 85 MB in int8). A failed load is tried again at the next call.
func (m *Model) Text() (*Text, error) {
	m.textMu.Lock()
	defer m.textMu.Unlock()
	if m.text != nil {
		return m.text, nil
	}
	t, err := loadText(m.Cfg, weights{f: m.file}, m.quantized)
	if err != nil {
		return nil, err
	}
	t.enc.workers = m.workers
	m.text = t
	return t, nil
}

// Load reads a model directory (config.json, model.safetensors). int8
// computes the layer products in int8 (see linalg.MatMul8).
func Load(dir string, quantized bool) (*Model, error) {
	cfg, err := ReadConfig(dir)
	if err != nil {
		return nil, err
	}
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	w := weights{f: f}
	m := &Model{Cfg: cfg, file: f, quantized: quantized, workers: 1}
	if m.Vision, err = loadVision(cfg, w, quantized); err != nil {
		return nil, err
	}
	scale, err := w.get("logit_scale", 1)
	if err != nil {
		return nil, err
	}
	bias, err := w.get("logit_bias", 1)
	if err != nil {
		return nil, err
	}
	m.LogitScale, m.LogitBias = scale[0], bias[0]
	return m, nil
}

// SetThreads bounds the cores one image or one text uses (default 1; 0:
// all). An image is 64 rows: on a hybrid processor, spreading it beyond
// the performance cores slows it down (48 ms on two performance cores, 73
// on one, 160 on all fourteen of a Core Ultra 7 265U). Not safe to call
// during a computation; the text tower takes it when it loads.
func (m *Model) SetThreads(n int) {
	if n <= 0 {
		n = linalg.Workers()
	}
	m.workers, m.Vision.enc.workers = n, n
}

// Logit turns the cosine between an image and a text embedding into the
// model's logit; its sigmoid is the probability that the text describes
// the image.
func (m *Model) Logit(cos float32) float32 {
	return cos*float32(math.Exp(float64(m.LogitScale))) + m.LogitBias
}

// weights reads named tensors and checks their shapes.
type weights struct{ f *safetensors.File }

func (w weights) get(name string, shape ...int) ([]float32, error) {
	t, ok, err := w.f.Tensor(name)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("siglip: tensor %s missing", name)
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(t.Data) != n {
		return nil, fmt.Errorf("siglip: %s: shape %v, expected %v", name, t.Shape, shape)
	}
	w.f.Evict(name)
	return t.Data, nil
}
