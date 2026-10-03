package indecis

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/modernbert"
	"github.com/bornholm/indecis/internal/safetensors"
	"github.com/bornholm/indecis/tokenizer"
)

// Answer is the answer to a question.
type Answer struct {
	Question string `json:"question"`
	Kind     Kind   `json:"kind"`
	// P is the probability that the answer is "true" (Noul).
	P float64 `json:"p,omitempty"`
	// Choice is the most probable option (Choice), or the most probable
	// level (Score).
	Choice string `json:"choice,omitempty"`
	// Score is the expected level, between 0 and the number of levels
	// minus one.
	Score float64 `json:"score,omitempty"`
	// Probs is the distribution over the options or the levels.
	Probs map[string]float64 `json:"probs,omitempty"`
	// Confidence is the probability of the chosen answer.
	Confidence float64 `json:"confidence"`
	// Margin (Choice) is the gap between the chosen probability and the
	// average of the others, CLM's confidence: close to 0 when the
	// options are equally likely, even if there are many of them.
	Margin float64 `json:"margin,omitempty"`
	// Spans lists the passages found (Spans), in text order.
	Spans []Span `json:"spans,omitempty"`
}

// Decision groups a text's answers, by question name.
type Decision map[string]Answer

// Info describes a model's origin.
type Info struct {
	Backbone string `json:"backbone,omitempty"`
	// TrainPrior is, for each Noul question, the proportion of "true"
	// answers in the training corpus. This is what is needed to correct
	// the probability when the proportion differs in production (see
	// calibrate.PriorShift).
	TrainPrior map[string]float64 `json:"train_prior,omitempty"`
	Steps      int                `json:"steps,omitempty"`
	// EmbedScale multiplies the cosines in ChooseNearest (0: the default
	// value).
	EmbedScale float64 `json:"embed_scale,omitempty"`
}

// Model is a decision model: an encoder and one head per question.
type Model struct {
	schema Schema
	enc    *modernbert.Model
	tok    *tokenizer.Tokenizer
	// tokenizerPath is the original tokenizer.json, copied by Save.
	// Keeping it in memory would cost 34 MB for a rare use.
	tokenizerPath string
	// tokenizerJSON replaces tokenizerPath when the tokenizer has been
	// modified (PruneVocabulary).
	tokenizerJSON []byte
	heads         []*head
	temps         []float64
	maxLen        int
	paired        bool
	info          Info
	spanBias      map[string]float64 // see WithSpanBias
	embedCache    *lru               // see WithEmbedCache
	embedInt8     bool               // see WithInt8Embeddings
	batcher       *batcher           // see WithBatching
}

// Input is a text to judge and its optional context.
type Input struct {
	Context string
	Text    string
}

// Construction options.
type Option func(*Model)

// WithMaxLen sets the maximum length in tokens; longer texts are
// truncated. 256 by default.
func WithMaxLen(n int) Option { return func(m *Model) { m.maxLen = n } }

// WithThreads bounds the number of cores used by the computations (0:
// all). The setting applies to the whole process. At inference, a batch
// of fewer than 1024 positions is computed on a single core anyway, the
// fastest for one sentence; long texts and large batches spread out.
func WithThreads(n int) Option { return func(*Model) { linalg.SetMaxWorkers(n) } }

// WithInt8 makes the encoder's layers compute in int8 (per-channel
// weights, per-token activations) when the processor has AVX-VNNI;
// without it, the option has no effect. Inference is about twice as fast
// and the layer weights four times smaller; the decisions of the
// injection detector (xolo-plugin-injection-guard) are unchanged on its
// reference set. The gap should still be checked for each model (Evaluate
// with and without the option).
func WithInt8() Option {
	return func(m *Model) { m.enc.SetInt8(linalg.Int8Fast()) }
}

// WithEmbedCache keeps in memory the embeddings of the n most recent
// distinct texts passed to Embed, and of the options prepared by
// PrepareCandidates (so ChooseNearest, DecideOpen): a reused option is not
// re-encoded. Texts judged by ChooseIn are not kept. The cache is cleared
// on every training run.
func WithEmbedCache(n int) Option {
	return func(m *Model) {
		if n > 0 {
			m.embedCache = newLRU(n)
		}
	}
}

// WithBatching groups the computations of simultaneous requests: workers
// goroutines (0: one per core) each take the first pending request and all
// those already queued, up to maxRows positions (0: 512), and compute
// them in a single pass. Without load, a request leaves right away,
// without waiting; under load, the batches grow on their own and the
// larger matrix products are more efficient. The sequences of a batch are
// laid end to end without padding.
//
// On CPU, the gain is small: each request already occupies a core, and
// requests of 100 to 256 tokens already make fairly large matrix
// products. Measured on the decision server (32 clients): a few % of
// throughput on single-sentence requests, nothing beyond that, for more
// memory.
func WithBatching(workers, maxRows int) Option {
	return func(m *Model) {
		if workers <= 0 {
			workers = runtime.GOMAXPROCS(0)
		}
		if maxRows <= 0 {
			maxRows = 512
		}
		m.batcher = newBatcher(m, workers, maxRows)
	}
}

// WithInt8Embeddings makes Save write the embedding table in int8, one
// scale per row: half the size of bf16, on disk as well as in the pages
// read. Rows modified by training lose their exact value; the gap is
// measured with Evaluate before and after. A model saved this way is
// recognized on load and stays that way. Hugging Face tools do not read
// this format.
func WithInt8Embeddings() Option { return func(m *Model) { m.embedInt8 = true } }

// WithSpanBias favors passages in the decoding of a Spans question:
// bias is added to the log-probability of every passage tag against the
// outside tag. A positive bias finds more passages and longer ones, more
// recall for less precision, the trade-off masking personal data asks
// for; it plays the part of a heavier loss weight on passages, without
// training again. Confidences stay those of the model.
func WithSpanBias(question string, bias float64) Option {
	return func(m *Model) {
		if m.spanBias == nil {
			m.spanBias = map[string]float64{}
		}
		m.spanBias[question] = bias
	}
}

// SetSpanBias changes the decoding bias of a Spans question (see
// WithSpanBias). It must not run during a decision.
func (m *Model) SetSpanBias(question string, bias float64) {
	WithSpanBias(question, bias)(m)
}

// WithPairs makes the model read pairs (context, text): the system prompt
// and the message, for example. All inputs are then encoded as a pair,
// including an empty context, so that training and inference see the same
// shape. See tokenizer.EncodePair for truncation.
func WithPairs() Option { return func(m *Model) { m.paired = true } }

// Paired indicates whether the model reads pairs (context, text).
func (m *Model) Paired() bool { return m.paired }

// tokenize encodes an input according to the model's mode.
func (m *Model) tokenize(context, text string) ([]int32, error) {
	if m.paired {
		return m.tok.EncodePair(context, text, m.maxLen), nil
	}
	if context != "" {
		return nil, fmt.Errorf("indecis: context given to a model built without WithPairs")
	}
	return m.tok.EncodeMax(text, m.maxLen), nil
}

// TokenIDs returns the ids of the tokens the model reads for an input,
// truncation included.
func (m *Model) TokenIDs(in Input) ([]int32, error) {
	return m.tokenize(in.Context, in.Text)
}

// Tokens returns the number of tokens an input occupies, truncation
// included: what the model actually reads.
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

// New creates an untrained model from a backbone in transformers format
// (config.json, model.safetensors, tokenizer.json). The heads are
// randomly initialized with seed.
func New(backboneDir string, schema Schema, seed int64, opts ...Option) (*Model, error) {
	if err := schema.Validate(); err != nil {
		return nil, err
	}
	enc, err := modernbert.Load(backboneDir)
	if err != nil {
		return nil, err
	}
	tokPath := filepath.Join(backboneDir, "tokenizer.json")
	tok, err := tokenizer.LoadShared(tokPath)
	if err != nil {
		return nil, err
	}
	m := &Model{
		schema: schema, enc: enc, tok: tok, tokenizerPath: tokPath,
		maxLen: 256, info: Info{Backbone: filepath.Base(backboneDir)},
	}
	for _, o := range opts {
		o(m)
	}
	if m.paired && m.hasSpans() {
		return nil, fmt.Errorf("indecis: Spans questions do not read pairs (WithPairs)")
	}
	rng := rand.New(rand.NewSource(seed))
	for _, q := range schema {
		m.heads = append(m.heads, newHead(q, enc.Cfg.Hidden, rng))
		m.temps = append(m.temps, 1)
	}
	return m, nil
}

// Schema returns the model's schema.
func (m *Model) Schema() Schema { return m.schema }

// Info returns the model's origin.
func (m *Model) Info() Info { return m.info }

// Temperatures returns the temperature of each question.
func (m *Model) Temperatures() map[string]float64 {
	out := map[string]float64{}
	for i, q := range m.schema {
		out[q.Name] = m.temps[i]
	}
	return out
}

// Decide answers all the questions for each text, without context.
func (m *Model) Decide(ctx context.Context, texts ...string) ([]Decision, error) {
	return m.DecideInputs(ctx, textsToInputs(texts)...)
}

// DecideInputs answers all the questions for each input.
func (m *Model) DecideInputs(ctx context.Context, inputs ...Input) ([]Decision, error) {
	res, err := m.infer(ctx, inputs, 32)
	if err != nil {
		return nil, err
	}
	out := make([]Decision, len(inputs))
	for i := range inputs {
		d := Decision{}
		for qi, h := range m.heads {
			if h.q.Kind == Spans {
				d[h.q.Name] = m.answerSpans(qi, inputs[i].Text, res[i])
				continue
			}
			d[h.q.Name] = h.answer(res[i].pooled[qi], m.temps[qi])
		}
		out[i] = d
	}
	return out, nil
}

// Logits returns, for each text, the raw logits of each question, before
// temperature: a caller that applies its own calibration or its own prior
// correction starts from there (see Temperatures and Info). Spans
// questions are left out.
func (m *Model) Logits(ctx context.Context, texts ...string) ([]map[string][]float64, error) {
	return m.LogitsInputs(ctx, textsToInputs(texts)...)
}

// LogitsInputs is Logits for inputs with context.
func (m *Model) LogitsInputs(ctx context.Context, inputs ...Input) ([]map[string][]float64, error) {
	raw, err := m.logits(ctx, inputs, 32)
	if err != nil {
		return nil, err
	}
	out := make([]map[string][]float64, len(inputs))
	for i := range raw {
		out[i] = make(map[string][]float64, len(m.heads))
		for qi, h := range m.heads {
			if h.q.Kind != Spans {
				out[i][h.q.Name] = raw[i][qi]
			}
		}
	}
	return out, nil
}

// logits returns, for each text, the raw logits of each question (nil
// for Spans questions).
func (m *Model) logits(ctx context.Context, inputs []Input, batchSize int) ([][][]float64, error) {
	res, err := m.infer(ctx, inputs, batchSize)
	if err != nil {
		return nil, err
	}
	out := make([][][]float64, len(inputs))
	for i, r := range res {
		out[i] = r.pooled
	}
	return out, nil
}

func (m *Model) tokenizeAll(inputs []Input) ([][]int32, error) {
	ids := make([][]int32, len(inputs))
	for i, in := range inputs {
		var err error
		if ids[i], err = m.tokenize(in.Context, in.Text); err != nil {
			return nil, err
		}
	}
	return ids, nil
}

// forEachPooled encodes token sequences and passes fn the mean embedding
// of each one (reused buffer: fn copies it if it keeps it). Sequences are
// grouped by length to limit padding.
func (m *Model) forEachPooled(ctx context.Context, ids [][]int32, batchSize int, fn func(i int, x []float32)) error {
	order := make([]int, len(ids))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) < len(ids[order[b]]) })

	H := m.enc.Cfg.Hidden
	for start := 0; start < len(order); start += batchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		idx := order[start:min(start+batchSize, len(order))]
		seqs := make([][]int32, len(idx))
		for j, i := range idx {
			seqs[j] = ids[i]
		}
		pooled, err := m.encodeSeqs(ctx, seqs)
		if err != nil {
			return err
		}
		for j, i := range idx {
			fn(i, pooled[j*H:(j+1)*H])
		}
	}
	return nil
}

// encodeSeqs computes the mean embeddings of sequences, either directly or
// through WithBatching's grouping.
func (m *Model) encodeSeqs(ctx context.Context, seqs [][]int32) ([]float32, error) {
	if m.batcher != nil {
		return m.batcher.encode(ctx, seqs)
	}
	return m.enc.EncodeSeqs(seqs)
}

// Files of a saved model.
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

// Tensors specific to indecis in model.safetensors: the embedding rows
// modified by training, in float32.
const (
	exactRows   = "indecis.embeddings.exact_rows"
	exactValues = "indecis.embeddings.exact_values"
	embName     = "embeddings.tok_embeddings.weight"
	embScale    = "indecis.embeddings.scale"
)

// lazyMatrix recognizes the layer matrices, read on demand.
func lazyMatrix(name string) bool {
	for _, suffix := range []string{".attn.Wqkv.weight", ".attn.Wo.weight", ".mlp.Wi.weight", ".mlp.Wo.weight"} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// quantizeRows quantizes w (rows x h) to int8 per row: q = round(w / s),
// s = max |w| / 127. The integers are returned as float32 for
// safetensors.Write.
func quantizeRows(w []float32, rows, h int) (q, scale []float32) {
	q = make([]float32, rows*h)
	scale = make([]float32, rows)
	for r := 0; r < rows; r++ {
		row := w[r*h : (r+1)*h]
		var mx float32
		for _, v := range row {
			mx = max(mx, float32(math.Abs(float64(v))))
		}
		if mx == 0 {
			continue
		}
		s := mx / 127
		scale[r] = s
		for j, v := range row {
			q[r*h+j] = float32(math.RoundToEven(float64(v / s)))
		}
	}
	return q, scale
}

// Save writes the model to dir. The directory stays readable by
// transformers: config.json and the encoder's weights keep their names.
func (m *Model) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tensors := map[string]safetensors.Tensor{}
	for _, p := range m.enc.Params() {
		tensors[p.Name] = safetensors.Tensor{Shape: p.Shape, Data: p.W}
	}
	H := m.enc.Cfg.Hidden
	// The embedding table (93% of the weights) is written in bf16, the
	// backbone's original format. The rows that training modified are no
	// longer exactly representable: they are added in float32, so that
	// Load gives back the model bit for bit.
	emb := m.enc.Emb
	embW := m.enc.EmbeddingMatrix()
	if m.embedInt8 {
		q, scale := quantizeRows(embW, emb.Shape[0], H)
		tensors[emb.Name] = safetensors.Tensor{Shape: emb.Shape, Data: q, DType: "I8"}
		tensors[embScale] = safetensors.Tensor{Shape: []int{emb.Shape[0]}, Data: scale}
		embW = nil // no exact rows: the table is quantized
	} else {
		tensors[emb.Name] = safetensors.Tensor{Shape: emb.Shape, Data: embW, DType: "BF16"}
	}
	var rows, values []float32
	for r := 0; embW != nil && r < emb.Shape[0]; r++ {
		row := embW[r*H : (r+1)*H]
		for _, v := range row {
			if safetensors.FromBF16(safetensors.ToBF16(v)) != v {
				rows = append(rows, float32(r)) // exact: r < 2^24
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
	// Each file is written alongside then renamed: a model loaded from dir
	// reads its weights in the memory-mapped file, which must not be
	// rewritten in place.
	err := writeFile(filepath.Join(dir, fileWeights), func(w io.Writer) error {
		return safetensors.Write(w, tensors, map[string]string{"format": "pt"})
	})
	if err != nil {
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
	if err := writeBytes(filepath.Join(dir, fileConfig), cfgJSON); err != nil {
		return err
	}
	tokRaw, err := m.tokenizerSource()
	if err != nil {
		return err
	}
	if err := writeBytes(filepath.Join(dir, fileTokenizer), tokRaw); err != nil {
		return err
	}
	meta, err := json.MarshalIndent(metaJSON{
		Format: formatVersion, Schema: m.schema, Temperatures: m.Temperatures(), MaxLen: m.maxLen, Paired: m.paired, Info: m.info,
	}, "", "  ")
	if err != nil {
		return err
	}
	return writeBytes(filepath.Join(dir, fileMeta), meta)
}

// writeFile writes path through a temporary file renamed afterward.
func writeFile(path string, write func(io.Writer) error) error {
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name()) // no effect after the rename
	bw := bufio.NewWriterSize(f, 1<<20)
	if err := write(bw); err != nil {
		f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Chmod(0o644); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func writeBytes(path string, b []byte) error {
	return writeFile(path, func(w io.Writer) error {
		_, err := w.Write(b)
		return err
	})
}

// Open loads a model written by Save, or, if dir does not contain one (no
// indecis.json), a backbone in transformers format: it then only answers
// open questions (ChooseNearest, DecideOpen), without training. Its
// learned question, "match", is only an untrained anchor point.
func Open(dir string, opts ...Option) (*Model, error) {
	if _, err := os.Stat(filepath.Join(dir, fileMeta)); err == nil {
		return Load(dir, opts...)
	}
	return New(dir, Schema{NewNoul("match", "")}, 1, opts...)
}

// Load reads a model written by Save.
func Load(dir string, opts ...Option) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, fileMeta))
	if err != nil {
		return nil, err
	}
	var meta metaJSON
	if err := json.Unmarshal(b, &meta); err != nil {
		return nil, fmt.Errorf("indecis: %s: %w", fileMeta, err)
	}
	if meta.Format != formatVersion {
		return nil, fmt.Errorf("indecis: unsupported format %q", meta.Format)
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
	// The weights are memory-mapped. The embedding table, written in bf16
	// by Save, is read in place; the rest is decoded to float32.
	f, err := safetensors.Open(filepath.Join(dir, fileWeights))
	if err != nil {
		return nil, err
	}
	tensors := map[string]safetensors.Tensor{}
	var table *mappedEmbeddings
	embedInt8 := false
	for _, name := range f.Names() {
		if name == exactRows || name == exactValues || name == embScale {
			continue
		}
		if dtype, shape, raw, _ := f.Raw(name); name == embName && dtype == "I8" && len(shape) == 2 && shape[1] == cfg.Hidden {
			sc, ok, err := f.Tensor(embScale)
			if err != nil || !ok {
				return nil, fmt.Errorf("indecis: int8 table without scales (%v)", err)
			}
			if table, err = newInt8Embeddings(cfg.Hidden, raw, sc.Data); err != nil {
				return nil, err
			}
			tensors[name] = safetensors.Tensor{Shape: shape}
			embedInt8 = true
			continue
		}
		if dtype, shape, raw, _ := f.Raw(name); name == embName && dtype == "BF16" && len(shape) == 2 && shape[1] == cfg.Hidden {
			rows, _, err := f.Tensor(exactRows)
			if err != nil {
				return nil, err
			}
			_, _, values, _ := f.Raw(exactValues)
			if table, err = newMappedEmbeddings(cfg.Hidden, raw, rows.Data, values); err != nil {
				return nil, err
			}
			tensors[name] = safetensors.Tensor{Shape: shape}
			continue
		}
		if lazyMatrix(name) {
			// The layer matrices are read when preparing for inference,
			// one at a time (see SetCompact below).
			_, shape, _, _ := f.Raw(name)
			tensors[name] = safetensors.Tensor{Shape: shape}
			continue
		}
		t, _, err := f.Tensor(name)
		if err != nil {
			return nil, err
		}
		tensors[name] = t
		f.Evict(name) // the decoded copy is enough
	}
	enc, err := modernbert.FromTensors(cfg, tensors)
	if err != nil {
		return nil, err
	}
	// A loaded model is first used for inference: its matrices are only
	// kept packed (see modernbert.SetCompact).
	enc.SetCompact(func(name string) ([]float32, error) {
		t, ok, err := f.Tensor(name)
		if err == nil && !ok {
			err = fmt.Errorf("tensor %s missing", name)
		}
		f.Evict(name)
		return t.Data, err
	})
	if table != nil {
		enc.SetEmbeddingTable(table)
	} else if rows, ok, _ := f.Tensor(exactRows); ok {
		values, _, err := f.Tensor(exactValues)
		if err != nil {
			return nil, err
		}
		H := cfg.Hidden
		if len(values.Data) != len(rows.Data)*H {
			return nil, fmt.Errorf("indecis: malformed exact rows")
		}
		for i, r := range rows.Data {
			id := int(r)
			if id < 0 || id >= cfg.Vocab {
				return nil, fmt.Errorf("indecis: exact row %d out of vocabulary", id)
			}
			copy(enc.Emb.W[id*H:(id+1)*H], values.Data[i*H:(i+1)*H])
		}
	}
	tokPath := filepath.Join(dir, fileTokenizer)
	tok, err := tokenizer.LoadShared(tokPath)
	if err != nil {
		return nil, err
	}
	m := &Model{schema: meta.Schema, enc: enc, tok: tok, tokenizerPath: tokPath, embedInt8: embedInt8, maxLen: meta.MaxLen, paired: meta.Paired, info: meta.Info}
	H := cfg.Hidden
	for _, q := range meta.Schema {
		h := newHead(q, H, rand.New(rand.NewSource(0)))
		w, okW := tensors["heads."+q.Name+".weight"]
		bb, okB := tensors["heads."+q.Name+".bias"]
		if !okW || !okB || len(w.Data) != len(h.w) || len(bb.Data) != len(h.b) {
			return nil, fmt.Errorf("indecis: head %s missing or malformed", q.Name)
		}
		h.w, h.b = w.Data, bb.Data
		m.heads = append(m.heads, h)
		t := meta.Temperatures[q.Name]
		if t <= 0 {
			t = 1
		}
		m.temps = append(m.temps, t)
	}
	for _, o := range opts {
		o(m)
	}
	return m, nil
}

func (m *Model) tokenizerSource() ([]byte, error) {
	if m.tokenizerJSON != nil {
		return m.tokenizerJSON, nil
	}
	b, err := os.ReadFile(m.tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("indecis: original tokenizer: %w", err)
	}
	return b, nil
}
