// Package ocr reads printed text lines from images, with a small recognizer
// that indecis trains and runs on CPU.
//
//	r, _ := ocr.Load(dir)
//	page := ocr.Gray(img)
//	line := r.ReadLine(page, ocr.Box{X0: 120, Y0: 340, X1: 980, Y1: 372})
//	line.Text
//
// A line is given by its ink box on the page; finding the lines of a page
// is not done here yet. A line image on its own (from a dataset) is read
// with ReadLine(page, InkBox(page)).
package ocr

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/bornholm/indecis/internal/charlm"
	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linerec"
)

// Recognizer reads text lines. It is safe for concurrent use.
type Recognizer struct {
	m     *linerec.Model
	chars []string // symbol i+1's text as read; 0 is the CTC blank
	vocab []string // the model's characters, as trained
	crop  CropOptions
	lm    *charlm.Model // nil: greedy decoding only
	lmOpt LMOptions
	retry float32 // lines read less surely are read again (WithRetry)

	states sync.Pool // *linerec.State
}

// Vocab is a recognizer's symbol list (vocab.json).
type Vocab struct {
	Blank int      `json:"blank"` // always 0
	Chars []string `json:"chars"` // symbol i+1
}

// Option configures a recognizer.
type Option func(*options)

type options struct {
	int8   bool
	lm     *charlm.Model
	lmOpts LMOptions
	retry  float32
}

// WithInt8 computes the pointwise convolutions and the head in int8 when
// the processor has the int8 kernel (AVX-VNNI).
func WithInt8() Option { return func(o *options) { o.int8 = true } }

// WithRetry reads again the lines read less surely than below (their
// lowest step probability, Line.Confidence), from boxes a little
// shifted, grown and shrunk, and keeps the surest reading.
func WithRetry(below float32) Option { return func(o *options) { o.retry = below } }

// Load reads a recognizer from dir: config.json, model.safetensors and
// vocab.json.
func Load(dir string, opts ...Option) (*Recognizer, error) {
	m, err := linerec.Load(dir)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, err
	}
	var v Vocab
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, fmt.Errorf("ocr: vocab.json: %w", err)
	}
	return New(m, v, opts...)
}

// New wraps a line model and its vocabulary.
func New(m *linerec.Model, v Vocab, opts ...Option) (*Recognizer, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	if v.Blank != 0 {
		return nil, fmt.Errorf("ocr: blank must be symbol 0, not %d", v.Blank)
	}
	if len(v.Chars)+1 != m.Cfg.Vocab {
		return nil, fmt.Errorf("ocr: %d characters for %d symbols", len(v.Chars), m.Cfg.Vocab)
	}
	if o.int8 {
		m.PrepareInt8()
	} else {
		m.Prepare()
	}
	// Readings are written as references are normalized (NormalizeText):
	// a model trained before the micro sign became mu still has it.
	chars := slices.Clone(v.Chars)
	for i, c := range chars {
		if c == "µ" {
			chars[i] = "μ"
		}
	}
	return &Recognizer{m: m, chars: chars, vocab: v.Chars, crop: ModelCrop(m), lm: o.lm, lmOpt: o.lmOpts, retry: o.retry}, nil
}

// Model returns the line model. Training it (Train) drops its packed
// weights: call its Prepare afterwards to read at full speed again.
func (r *Recognizer) Model() *linerec.Model { return r.m }

// Vocab returns the recognizer's vocabulary.
func (r *Recognizer) Vocab() Vocab { return Vocab{Chars: r.vocab} }

// ModelCrop returns the crop a line model reads: the defaults, at the
// model's line height and window.
func ModelCrop(m *linerec.Model) CropOptions {
	crop := DefaultCrop()
	crop.Height, crop.Window = m.Cfg.Height, m.Cfg.Window
	return crop
}

// CropOptions returns the crop the recognizer reads lines with.
func (r *Recognizer) CropOptions() CropOptions { return r.crop }

// Char is a recognized character.
type Char struct {
	Text   string
	X0, X1 float64 // page x of its steps
	P      float32 // its probability, highest over its steps
}

// Line is a recognized line.
type Line struct {
	Text       string
	Chars      []Char
	Confidence float32 // lowest probability among the line's steps
}

// ReadLine reads the line in box.
func (r *Recognizer) ReadLine(page *image.Gray, box Box) Line {
	line := r.ReadCrop(CropLine(page, box, r.crop))
	if line.Confidence >= r.retry {
		return line
	}
	// A box a little off the line (a detector's, on a damaged page) cuts
	// or crowds it: read it again shifted, grown, shrunk, and keep the
	// surest reading.
	h := box.Y1 - box.Y0
	best, bestP := line, meanP(line)
	for _, d := range [][2]float64{{-0.12, -0.12}, {0.12, 0.12}, {-0.15, 0.15}, {0.1, -0.1}} {
		b := Box{X0: box.X0, Y0: box.Y0 + d[0]*h, X1: box.X1, Y1: box.Y1 + d[1]*h}
		l := r.ReadCrop(CropLine(page, b, r.crop))
		if p := meanP(l); p > bestP {
			best, bestP = l, p
		}
	}
	return best
}

// meanP is the mean probability of a line's characters (0 without any).
func meanP(l Line) float64 {
	if len(l.Chars) == 0 {
		return 0
	}
	s := 0.0
	for _, c := range l.Chars {
		s += float64(c.P)
	}
	return s / float64(len(l.Chars))
}

// ReadLines reads several lines of a page, in parallel.
func (r *Recognizer) ReadLines(page *image.Gray, boxes []Box) []Line {
	out := make([]Line, len(boxes))
	linalg.Parallel(len(boxes), 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = r.ReadLine(page, boxes[i])
		}
	})
	return out
}

// ReadCrop reads a line already cut by CropLine.
func (r *Recognizer) ReadCrop(c Crop) Line {
	s, _ := r.states.Get().(*linerec.State)
	if s == nil {
		s = new(linerec.State)
	}
	defer r.states.Put(s)
	r.m.Forward(s, c.Ink, c.W)
	V := r.m.Cfg.Vocab
	syms, conf := linerec.Greedy(s.Logits, s.L, V)
	if r.lm != nil && conf < r.lmOpt.Gate {
		syms = beamDecode(s.Logits, s.L, V, r.chars, r.lm, r.lmOpt)
		// A symbol's probability: its highest over its steps.
		for i := range syms {
			sy := &syms[i]
			for t := sy.Step; t < min(s.L, sy.Step+sy.Steps); t++ {
				row := s.Logits[t*V : (t+1)*V]
				sy.P = max(sy.P, softmaxAt(row, sy.ID))
			}
		}
	}
	line := Line{Confidence: conf, Chars: make([]Char, len(syms))}
	var sb strings.Builder
	S := float64(r.m.Cfg.Stride)
	for i, sy := range syms {
		t := r.chars[sy.ID-1]
		sb.WriteString(t)
		line.Chars[i] = Char{
			Text: t,
			X0:   c.PageX(float64(sy.Step) * S),
			X1:   c.PageX(float64(sy.Step+sy.Steps) * S),
			P:    sy.P,
		}
	}
	line.Text = sb.String()
	return line
}

// softmaxAt is the softmax probability of row[i].
func softmaxAt(row []float32, i int) float32 {
	mx := row[0]
	for _, v := range row {
		mx = max(mx, v)
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v - mx))
	}
	return float32(math.Exp(float64(row[i]-mx)) / sum)
}
