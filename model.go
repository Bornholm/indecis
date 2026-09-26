package indecis

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/internal/modernbert"
	"github.com/bornholm/indecis/internal/safetensors"
	"github.com/bornholm/indecis/tokenizer"
)

// Answer est la réponse à une question.
type Answer struct {
	Question string `json:"question"`
	Kind     Kind   `json:"kind"`
	// P est la probabilité que la réponse soit « vrai » (Noul).
	P float64 `json:"p,omitempty"`
	// Choice est l'option la plus probable (Choice), ou le niveau le plus
	// probable (Score).
	Choice string `json:"choice,omitempty"`
	// Score est le niveau attendu, entre 0 et le nombre de niveaux moins un.
	Score float64 `json:"score,omitempty"`
	// Probs est la distribution sur les options ou les niveaux.
	Probs map[string]float64 `json:"probs,omitempty"`
	// Confidence est la probabilité de la réponse retenue.
	Confidence float64 `json:"confidence"`
}

// Decision regroupe les réponses d'un texte, par nom de question.
type Decision map[string]Answer

// Info décrit l'origine d'un modèle.
type Info struct {
	Backbone string `json:"backbone,omitempty"`
	// TrainPrior est, pour chaque question Noul, la proportion de réponses
	// « vrai » dans le corpus d'entraînement. C'est ce qu'il faut pour
	// corriger la probabilité quand la proportion diffère en production
	// (voir calibrate.PriorShift).
	TrainPrior map[string]float64 `json:"train_prior,omitempty"`
	Steps      int                `json:"steps,omitempty"`
}

// Model est un modèle de décision : un encodeur et une tête par question.
type Model struct {
	schema       Schema
	enc          *modernbert.Model
	tok          *tokenizer.Tokenizer
	tokenizerRaw []byte
	heads        []*head
	temps        []float64
	maxLen       int
	paired       bool
	info         Info
}

// Input est un texte à juger et son contexte éventuel.
type Input struct {
	Context string
	Text    string
}

// Options de construction.
type Option func(*Model)

// WithMaxLen fixe la longueur maximale en tokens ; les textes plus longs sont
// tronqués. 256 par défaut.
func WithMaxLen(n int) Option { return func(m *Model) { m.maxLen = n } }

// WithPairs fait lire au modèle des paires (contexte, texte) : le prompt
// système et le message, par exemple. Toutes les entrées sont alors encodées
// en paire, contexte vide compris, pour que l'entraînement et l'inférence
// voient la même forme. Voir tokenizer.EncodePair pour la troncature.
func WithPairs() Option { return func(m *Model) { m.paired = true } }

// Paired indique si le modèle lit des paires (contexte, texte).
func (m *Model) Paired() bool { return m.paired }

// tokenize encode une entrée selon le mode du modèle.
func (m *Model) tokenize(context, text string) ([]int32, error) {
	if m.paired {
		return m.tok.EncodePair(context, text, m.maxLen), nil
	}
	if context != "" {
		return nil, fmt.Errorf("indecis: contexte fourni à un modèle construit sans WithPairs")
	}
	return m.tok.EncodeMax(text, m.maxLen), nil
}

// Tokens retourne le nombre de tokens qu'une entrée occupe, troncature
// comprise : ce que le modèle lit réellement.
func (m *Model) Tokens(in Input) (int, error) {
	ids, err := m.tokenize(in.Context, in.Text)
	return len(ids), err
}

func textsToInputs(texts []string) []Input {
	in := make([]Input, len(texts))
	for i, t := range texts {
		in[i].Text = t
	}
	return in
}

func examplesToInputs(examples []dataset.Example) []Input {
	in := make([]Input, len(examples))
	for i, e := range examples {
		in[i] = Input{Context: e.Context, Text: e.Text}
	}
	return in
}

// New crée un modèle non entraîné à partir d'un backbone au format
// transformers (config.json, model.safetensors, tokenizer.json). Les têtes
// sont initialisées aléatoirement avec la graine seed.
func New(backboneDir string, schema Schema, seed int64, opts ...Option) (*Model, error) {
	if err := schema.Validate(); err != nil {
		return nil, err
	}
	enc, err := modernbert.Load(backboneDir)
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(filepath.Join(backboneDir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Parse(raw)
	if err != nil {
		return nil, err
	}
	m := &Model{
		schema: schema, enc: enc, tok: tok, tokenizerRaw: raw,
		maxLen: 256, info: Info{Backbone: filepath.Base(backboneDir)},
	}
	for _, o := range opts {
		o(m)
	}
	rng := rand.New(rand.NewSource(seed))
	for _, q := range schema {
		m.heads = append(m.heads, newHead(q, enc.Cfg.Hidden, rng))
		m.temps = append(m.temps, 1)
	}
	return m, nil
}

// Schema retourne le schéma du modèle.
func (m *Model) Schema() Schema { return m.schema }

// Info retourne l'origine du modèle.
func (m *Model) Info() Info { return m.info }

// Temperatures retourne la température de chaque question.
func (m *Model) Temperatures() map[string]float64 {
	out := map[string]float64{}
	for i, q := range m.schema {
		out[q.Name] = m.temps[i]
	}
	return out
}

// Decide répond à toutes les questions pour chaque texte, sans contexte.
func (m *Model) Decide(ctx context.Context, texts ...string) ([]Decision, error) {
	return m.DecideInputs(ctx, textsToInputs(texts)...)
}

// DecideInputs répond à toutes les questions pour chaque entrée.
func (m *Model) DecideInputs(ctx context.Context, inputs ...Input) ([]Decision, error) {
	logits, err := m.logits(ctx, inputs, 32)
	if err != nil {
		return nil, err
	}
	out := make([]Decision, len(inputs))
	for i := range inputs {
		d := Decision{}
		for qi, h := range m.heads {
			d[h.q.Name] = h.answer(logits[i][qi], m.temps[qi])
		}
		out[i] = d
	}
	return out, nil
}

// Logits retourne, pour chaque texte, les logits bruts de chaque question,
// avant température : un appelant qui applique sa propre calibration ou sa
// propre correction de prior part de là (voir Temperatures et Info).
func (m *Model) Logits(ctx context.Context, texts ...string) ([]map[string][]float64, error) {
	return m.LogitsInputs(ctx, textsToInputs(texts)...)
}

// LogitsInputs est Logits pour des entrées avec contexte.
func (m *Model) LogitsInputs(ctx context.Context, inputs ...Input) ([]map[string][]float64, error) {
	raw, err := m.logits(ctx, inputs, 32)
	if err != nil {
		return nil, err
	}
	out := make([]map[string][]float64, len(inputs))
	for i := range raw {
		out[i] = make(map[string][]float64, len(m.heads))
		for qi, h := range m.heads {
			out[i][h.q.Name] = raw[i][qi]
		}
	}
	return out, nil
}

// logits retourne, pour chaque texte, les logits bruts de chaque question.
// Les textes sont regroupés par longueur pour limiter le padding.
func (m *Model) logits(ctx context.Context, inputs []Input, batchSize int) ([][][]float64, error) {
	ids := make([][]int32, len(inputs))
	for i, in := range inputs {
		var err error
		if ids[i], err = m.tokenize(in.Context, in.Text); err != nil {
			return nil, err
		}
	}
	texts := inputs
	order := make([]int, len(texts))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) < len(ids[order[b]]) })

	out := make([][][]float64, len(texts))
	H := m.enc.Cfg.Hidden
	for start := 0; start < len(order); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		idx := order[start:min(start+batchSize, len(order))]
		seqs := make([][]int32, len(idx))
		for j, i := range idx {
			seqs[j] = ids[i]
		}
		b := modernbert.NewBatch(seqs, m.enc.Cfg.PadID)
		s, err := m.enc.Forward(b)
		if err != nil {
			return nil, err
		}
		pooled := modernbert.MeanPool(s.Hidden, b, H)
		for j, i := range idx {
			x := pooled[j*H : (j+1)*H]
			out[i] = make([][]float64, len(m.heads))
			for qi, h := range m.heads {
				out[i][qi] = h.logits(x)
			}
		}
	}
	return out, nil
}

// Fichiers d'un modèle sauvegardé.
const (
	fileWeights   = "model.safetensors"
	fileConfig    = "config.json"
	fileTokenizer = "tokenizer.json"
	fileMeta      = "indecis.json"
)

type metaJSON struct {
	Format       string             `json:"format"`
	Schema       Schema             `json:"schema"`
	Temperatures map[string]float64 `json:"temperatures"`
	MaxLen       int                `json:"max_len"`
	Paired       bool               `json:"paired,omitempty"`
	Info         Info               `json:"info"`
}

const formatVersion = "indecis/1"

// Tenseurs propres à indecis dans model.safetensors : les lignes
// d'embeddings modifiées par l'entraînement, en float32.
const (
	exactRows   = "indecis.embeddings.exact_rows"
	exactValues = "indecis.embeddings.exact_values"
)

// Save écrit le modèle dans dir. Le répertoire reste lisible par
// transformers : config.json et les poids de l'encodeur gardent leurs noms.
func (m *Model) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tensors := map[string]safetensors.Tensor{}
	for _, p := range m.enc.Params() {
		tensors[p.Name] = safetensors.Tensor{Shape: p.Shape, Data: p.W}
	}
	H := m.enc.Cfg.Hidden
	// La table d'embeddings (93 % des poids) est écrite en bf16, format
	// d'origine du backbone. Les lignes que l'entraînement a modifiées ne
	// sont plus représentables exactement : elles sont ajoutées en float32,
	// pour que Load redonne le modèle au bit près.
	emb := m.enc.Emb
	tensors[emb.Name] = safetensors.Tensor{Shape: emb.Shape, Data: emb.W, DType: "BF16"}
	var rows, values []float32
	for r := 0; r < emb.Shape[0]; r++ {
		row := emb.W[r*H : (r+1)*H]
		for _, v := range row {
			if safetensors.FromBF16(safetensors.ToBF16(v)) != v {
				rows = append(rows, float32(r)) // exact : r < 2^24
				values = append(values, row...)
				break
			}
		}
	}
	if len(rows) > 0 {
		tensors[exactRows] = safetensors.Tensor{Shape: []int{len(rows)}, Data: rows}
		tensors[exactValues] = safetensors.Tensor{Shape: []int{len(rows), H}, Data: values}
	}
	for _, h := range m.heads {
		tensors["heads."+h.q.Name+".weight"] = safetensors.Tensor{Shape: []int{h.rows, H}, Data: h.w}
		tensors["heads."+h.q.Name+".bias"] = safetensors.Tensor{Shape: []int{h.outs}, Data: h.b}
	}
	f, err := os.Create(filepath.Join(dir, fileWeights))
	if err != nil {
		return err
	}
	if err := safetensors.Write(f, tensors, map[string]string{"format": "pt"}); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	cfg := m.enc.Cfg
	cfgJSON, err := json.MarshalIndent(map[string]any{
		"architectures": []string{"ModernBertModel"}, "model_type": "modernbert",
		"hidden_size": cfg.Hidden, "num_hidden_layers": cfg.Layers, "num_attention_heads": cfg.Heads,
		"intermediate_size": cfg.Intermediate, "vocab_size": cfg.Vocab, "norm_eps": cfg.NormEps,
		"global_attn_every_n_layers": cfg.GlobalEvery, "local_attention": cfg.LocalAttention,
		"global_rope_theta": cfg.GlobalTheta, "local_rope_theta": cfg.LocalTheta,
		"pad_token_id": cfg.PadID, "hidden_activation": cfg.Activation,
		"attention_bias": false, "mlp_bias": false, "norm_bias": false,
	}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, fileConfig), cfgJSON, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, fileTokenizer), m.tokenizerRaw, 0o644); err != nil {
		return err
	}
	meta, err := json.MarshalIndent(metaJSON{
		Format: formatVersion, Schema: m.schema, Temperatures: m.Temperatures(), MaxLen: m.maxLen, Paired: m.paired, Info: m.info,
	}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, fileMeta), meta, 0o644)
}

// Load lit un modèle écrit par Save.
func Load(dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, fileMeta))
	if err != nil {
		return nil, err
	}
	var meta metaJSON
	if err := json.Unmarshal(b, &meta); err != nil {
		return nil, fmt.Errorf("indecis: %s : %w", fileMeta, err)
	}
	if meta.Format != formatVersion {
		return nil, fmt.Errorf("indecis: format %q non pris en charge", meta.Format)
	}
	if err := meta.Schema.Validate(); err != nil {
		return nil, err
	}
	cb, err := os.ReadFile(filepath.Join(dir, fileConfig))
	if err != nil {
		return nil, err
	}
	var cfg modernbert.Config
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return nil, err
	}
	tensors, _, err := safetensors.ReadFile(filepath.Join(dir, fileWeights))
	if err != nil {
		return nil, err
	}
	enc, err := modernbert.FromTensors(cfg, tensors)
	if err != nil {
		return nil, err
	}
	if rows, ok := tensors[exactRows]; ok {
		values := tensors[exactValues]
		H := cfg.Hidden
		if len(values.Data) != len(rows.Data)*H {
			return nil, fmt.Errorf("indecis: lignes exactes mal formées")
		}
		for i, r := range rows.Data {
			id := int(r)
			if id < 0 || id >= cfg.Vocab {
				return nil, fmt.Errorf("indecis: ligne exacte %d hors vocabulaire", id)
			}
			copy(enc.Emb.W[id*H:(id+1)*H], values.Data[i*H:(i+1)*H])
		}
	}
	raw, err := os.ReadFile(filepath.Join(dir, fileTokenizer))
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Parse(raw)
	if err != nil {
		return nil, err
	}
	m := &Model{schema: meta.Schema, enc: enc, tok: tok, tokenizerRaw: raw, maxLen: meta.MaxLen, paired: meta.Paired, info: meta.Info}
	H := cfg.Hidden
	for _, q := range meta.Schema {
		h := newHead(q, H, rand.New(rand.NewSource(0)))
		w, okW := tensors["heads."+q.Name+".weight"]
		bb, okB := tensors["heads."+q.Name+".bias"]
		if !okW || !okB || len(w.Data) != len(h.w) || len(bb.Data) != len(h.b) {
			return nil, fmt.Errorf("indecis: tête %s absente ou mal formée", q.Name)
		}
		h.w, h.b = w.Data, bb.Data
		m.heads = append(m.heads, h)
		t := meta.Temperatures[q.Name]
		if t <= 0 {
			t = 1
		}
		m.temps = append(m.temps, t)
	}
	return m, nil
}
