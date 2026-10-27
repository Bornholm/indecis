package ocr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linerec"
	"github.com/bornholm/indecis/internal/optim"
	"github.com/bornholm/indecis/internal/safetensors"
)

// LineSource yields training lines, endlessly or until io.EOF.
type LineSource interface {
	Next() (ShardLine, error)
}

// TrainOptions sets a training run.
type TrainOptions struct {
	Steps       int     // optimizer steps
	Batch       int     // lines per step
	LR          float64 // peak learning rate
	Warmup      int     // steps of linear warmup, then cosine decay to 0
	WeightDecay float64
	Clip        float64 // global gradient norm (0: none)

	// Slots is the number of gradient accumulators a batch is split over,
	// line i into slot i mod Slots, summed in order: the run gives the same
	// model whatever the number of cores. Default 16.
	Slots int

	Eval      []ShardLine // validation lines, read every EvalEvery steps
	EvalEvery int
	Out       string // where the model is saved: at each evaluation the best one, or at the end
	LogEvery  int
	Log       io.Writer
}

// TrainResult sums up a run.
type TrainResult struct {
	Steps    int
	Lines    int
	Skipped  int // lines with characters outside the vocabulary or too short for their text
	BestCER  float64
	BestStep int
}

// Train trains m on the lines of src. m's weights change in place.
func Train(ctx context.Context, m *linerec.Model, v Vocab, src LineSource, o TrainOptions) (TrainResult, error) {
	if o.Slots <= 0 {
		o.Slots = 16
	}
	if o.Batch <= 0 || o.Steps <= 0 {
		return TrainResult{}, fmt.Errorf("ocr: Steps and Batch must be positive")
	}
	if len(v.Chars)+1 != m.Cfg.Vocab {
		return TrainResult{}, fmt.Errorf("ocr: %d characters for %d symbols", len(v.Chars), m.Cfg.Vocab)
	}
	if o.Log == nil {
		o.Log = io.Discard
	}
	m.Unprepare()
	enc := v.Encoder()
	V := m.Cfg.Vocab

	type slot struct {
		g     *linerec.Grads
		s     linerec.State
		d     []float32
		loss  float64
		lines int
	}
	slots := make([]*slot, o.Slots)
	for i := range slots {
		slots[i] = &slot{g: m.NewGrads()}
	}
	total := m.NewGrads()
	dense := m.Dense(total)
	cfg := optim.DefaultConfig(o.LR)
	cfg.WeightDecay = o.WeightDecay
	opt := optim.New(cfg)

	res := TrainResult{BestCER: math.Inf(1)}
	type item struct {
		l   ShardLine
		ids []int
	}
	batch := make([]item, 0, o.Batch)
	var logLoss float64
	var logLines int
	start := time.Now()
	for step := 1; step <= o.Steps; step++ {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		batch = batch[:0]
		for len(batch) < o.Batch {
			l, err := src.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return res, err
			}
			ids, _, ok := enc.Encode(l.Text)
			_, steps := m.Cfg.Steps(l.W)
			if !ok || len(ids) == 0 || steps < minSteps(ids) {
				res.Skipped++
				continue
			}
			batch = append(batch, item{l, ids})
		}
		if len(batch) == 0 {
			break
		}

		scale := float32(1 / float64(len(batch)))
		linalg.Parallel(len(slots), 1, func(lo, hi int) {
			for k := lo; k < hi; k++ {
				sl := slots[k]
				sl.g.Zero()
				sl.loss, sl.lines = 0, 0
				for i := k; i < len(batch); i += len(slots) {
					it := batch[i]
					m.Forward(&sl.s, it.l.Ink, it.l.W)
					L := sl.s.L
					sl.d = grow(sl.d, L*V)
					loss, ok := linerec.CTC(sl.s.Logits, L, V, it.ids, sl.d)
					if !ok {
						continue
					}
					linalg.Scale(sl.d[:L*V], scale)
					m.Backward(&sl.s, sl.d[:L*V], sl.g)
					sl.loss += loss
					sl.lines++
				}
			}
		})
		total.Zero()
		for _, sl := range slots {
			total.Add(sl.g)
			logLoss += sl.loss
			logLines += sl.lines
			res.Lines += sl.lines
		}
		norm := optim.ClipGradNorm(o.Clip, dense, nil)
		lr := schedule(o, step)
		opt.Step(lr, dense, nil)
		res.Steps = step

		if o.LogEvery > 0 && step%o.LogEvery == 0 {
			el := time.Since(start).Seconds()
			fmt.Fprintf(o.Log, "step %6d  loss %7.3f  |g| %6.2f  lr %.2e  %5.0f lines/s\n",
				step, logLoss/float64(max(1, logLines)), norm, lr, float64(res.Lines)/el)
			logLoss, logLines = 0, 0
		}
		last := step == o.Steps
		if len(o.Eval) > 0 && ((o.EvalEvery > 0 && step%o.EvalEvery == 0) || last) {
			e := Evaluate(m, v, o.Eval)
			fmt.Fprintf(o.Log, "step %6d  eval CER %.2f%%  WER %.2f%%  exact %.1f%%\n",
				step, 100*e.CER(), 100*e.WER(), 100*float64(e.Exact)/float64(max(1, e.Lines)))
			if e.CER() < res.BestCER {
				res.BestCER, res.BestStep = e.CER(), step
				if o.Out != "" {
					if err := SaveModel(o.Out, m, v); err != nil {
						return res, err
					}
				}
			}
		} else if last && o.Out != "" && len(o.Eval) == 0 {
			if err := SaveModel(o.Out, m, v); err != nil {
				return res, err
			}
		}
	}
	return res, nil
}

// minSteps is the fewest CTC steps that can hold ids: one per symbol, plus
// a blank between two equal symbols.
func minSteps(ids []int) int {
	n := len(ids)
	for i := 1; i < len(ids); i++ {
		if ids[i] == ids[i-1] {
			n++
		}
	}
	return n
}

// schedule is the learning rate at step: linear warmup, then cosine decay
// to zero.
func schedule(o TrainOptions, step int) float64 {
	if step <= o.Warmup {
		return o.LR * float64(step) / float64(max(1, o.Warmup))
	}
	p := float64(step-o.Warmup) / float64(max(1, o.Steps-o.Warmup))
	return o.LR * 0.5 * (1 + math.Cos(math.Pi*min(p, 1)))
}

// Evaluate reads the lines with m (greedy decoding) and measures the
// error rates against their texts.
func Evaluate(m *linerec.Model, v Vocab, lines []ShardLine) ErrorRate {
	got := make([]string, len(lines))
	var pool sync.Pool
	linalg.Parallel(len(lines), 1, func(lo, hi int) {
		s, _ := pool.Get().(*linerec.State)
		if s == nil {
			s = new(linerec.State)
		}
		defer pool.Put(s)
		for i := lo; i < hi; i++ {
			m.Forward(s, lines[i].Ink, lines[i].W)
			syms, _ := linerec.Greedy(s.Logits, s.L, m.Cfg.Vocab)
			var b []byte
			for _, sy := range syms {
				b = append(b, v.Chars[sy.ID-1]...)
			}
			got[i] = string(b)
		}
	})
	var e ErrorRate
	for i, l := range lines {
		e.Add(got[i], l.Text)
	}
	return e
}

// SaveModel writes a recognizer to dir: config.json, model.safetensors and
// vocab.json, each replaced at once.
func SaveModel(dir string, m *linerec.Model, v Vocab) error {
	return saveFiles(dir, m.Cfg, m.Tensors(), map[string]any{"vocab.json": v})
}

// saveFiles writes config.json, model.safetensors and the extra JSON
// files to dir, each written aside then renamed.
func saveFiles(dir string, cfg any, tensors map[string]safetensors.Tensor, extra ...map[string]any) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	write := func(name string, fn func(io.Writer) error) error {
		tmp, err := os.CreateTemp(dir, name+".tmp*")
		if err != nil {
			return err
		}
		if err := fn(tmp); err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
			return err
		}
		if err := tmp.Close(); err != nil {
			return err
		}
		return os.Rename(tmp.Name(), filepath.Join(dir, name))
	}
	jsonOf := func(x any) func(io.Writer) error {
		return func(w io.Writer) error {
			e := json.NewEncoder(w)
			e.SetIndent("", " ")
			return e.Encode(x)
		}
	}
	if err := write("config.json", jsonOf(cfg)); err != nil {
		return err
	}
	for _, files := range extra {
		for name, x := range files {
			if err := write(name, jsonOf(x)); err != nil {
				return err
			}
		}
	}
	return write("model.safetensors", func(w io.Writer) error {
		return safetensors.Write(w, tensors, nil)
	})
}

// MemorySource cycles through lines held in memory, shuffled at each
// pass.
type MemorySource struct {
	lines []ShardLine
	order []int
	pos   int
	rng   *rand.Rand
}

// NewMemorySource creates a source over lines.
func NewMemorySource(lines []ShardLine, seed int64) *MemorySource {
	return &MemorySource{lines: lines, rng: rand.New(rand.NewSource(seed))}
}

// Next returns the next line; it never runs out (io.EOF only when empty).
func (s *MemorySource) Next() (ShardLine, error) {
	if len(s.lines) == 0 {
		return ShardLine{}, io.EOF
	}
	if s.pos == len(s.order) {
		s.order = s.rng.Perm(len(s.lines))
		s.pos = 0
	}
	s.pos++
	return s.lines[s.order[s.pos-1]], nil
}

// ShardSource streams the lines of shard files in a random order: shards
// in a shuffled order, lines through a shuffle buffer, pass after pass.
type ShardSource struct {
	paths  []string
	buffer []ShardLine
	size   int
	rng    *rand.Rand
	order  []int
	next   int
	cur    *ShardReader
}

// NewShardSource creates a source over shard files with a shuffle buffer
// of size lines.
func NewShardSource(paths []string, size int, seed int64) (*ShardSource, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("ocr: no shard")
	}
	return &ShardSource{paths: paths, size: max(1, size), rng: rand.New(rand.NewSource(seed))}, nil
}

// Next returns the next line.
func (s *ShardSource) Next() (ShardLine, error) {
	for len(s.buffer) < s.size {
		l, err := s.read()
		if err != nil {
			return l, err
		}
		s.buffer = append(s.buffer, l)
	}
	i := s.rng.Intn(len(s.buffer))
	l := s.buffer[i]
	s.buffer[i] = s.buffer[len(s.buffer)-1]
	s.buffer = s.buffer[:len(s.buffer)-1]
	return l, nil
}

// read returns the next line of the current shard, opening the next shard
// when one ends.
func (s *ShardSource) read() (ShardLine, error) {
	for empty := 0; empty <= len(s.paths); {
		if s.cur == nil {
			if s.next == len(s.order) {
				s.order = s.rng.Perm(len(s.paths))
				s.next = 0
			}
			r, err := OpenShard(s.paths[s.order[s.next]])
			s.next++
			if err != nil {
				return ShardLine{}, err
			}
			s.cur = r
		}
		l, err := s.cur.Next()
		if err == nil {
			return l, nil
		}
		s.cur.Close()
		s.cur = nil
		if !errors.Is(err, io.EOF) {
			return l, err
		}
		empty++
	}
	return ShardLine{}, fmt.Errorf("ocr: the shards hold no line")
}

func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

// MixSource draws each line from one of several sources, at given rates:
// a small set of real lines can weigh a fixed share of the batches against
// a large synthetic one.
type MixSource struct {
	sources []LineSource
	cum     []float64
	rng     *rand.Rand
}

// NewMixSource mixes sources with weights (normalized).
func NewMixSource(sources []LineSource, weights []float64, seed int64) *MixSource {
	m := &MixSource{sources: sources, rng: rand.New(rand.NewSource(seed))}
	var total float64
	for _, w := range weights {
		total += w
	}
	acc := 0.0
	for _, w := range weights {
		acc += w / total
		m.cum = append(m.cum, acc)
	}
	return m
}

// Next returns a line of a source drawn at its rate.
func (m *MixSource) Next() (ShardLine, error) {
	x := m.rng.Float64()
	for i, c := range m.cum {
		if x < c || i == len(m.cum)-1 {
			return m.sources[i].Next()
		}
	}
	return ShardLine{}, io.EOF
}
