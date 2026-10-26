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
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linerec"
)

// Recognizer reads text lines. It is safe for concurrent use.
type Recognizer struct {
	m     *linerec.Model
	chars []string // symbol i+1; 0 is the CTC blank
	crop  CropOptions

	states sync.Pool // *linerec.State
}

// Vocab is a recognizer's symbol list (vocab.json).
type Vocab struct {
	Blank int      `json:"blank"` // always 0
	Chars []string `json:"chars"` // symbol i+1
}

// Load reads a recognizer from dir: config.json, model.safetensors and
// vocab.json.
func Load(dir string) (*Recognizer, error) {
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
	return New(m, v)
}

// New wraps a line model and its vocabulary.
func New(m *linerec.Model, v Vocab) (*Recognizer, error) {
	if v.Blank != 0 {
		return nil, fmt.Errorf("ocr: blank must be symbol 0, not %d", v.Blank)
	}
	if len(v.Chars)+1 != m.Cfg.Vocab {
		return nil, fmt.Errorf("ocr: %d characters for %d symbols", len(v.Chars), m.Cfg.Vocab)
	}
	m.Prepare()
	crop := DefaultCrop()
	crop.Height, crop.Window, crop.Multiple = m.Cfg.Height, m.Cfg.Window, 2*m.Cfg.Stride
	return &Recognizer{m: m, chars: v.Chars, crop: crop}, nil
}

// Model returns the line model. Training it (Train) drops its packed
// weights: call its Prepare afterwards to read at full speed again.
func (r *Recognizer) Model() *linerec.Model { return r.m }

// Vocab returns the recognizer's vocabulary.
func (r *Recognizer) Vocab() Vocab { return Vocab{Chars: r.chars} }

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
	return r.ReadCrop(CropLine(page, box, r.crop))
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
	syms, conf := linerec.Greedy(s.Logits, s.L, r.m.Cfg.Vocab)
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
