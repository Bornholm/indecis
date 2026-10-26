package ocr

import (
	"bufio"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg" // line and page images
	_ "image/png"
	"io"
	"os"
	"path/filepath"
)

// Sample is a labeled line (one JSONL row): an image, the line's box on it
// (absent: the image is the line, see InkBox) and its text.
type Sample struct {
	Image string            `json:"image"` // relative to the JSONL file
	Box   []float64         `json:"box,omitempty"`
	Text  string            `json:"text"`
	Meta  map[string]string `json:"meta,omitempty"`
}

// ReadSamples reads a JSONL file of samples; image paths become relative to
// the working directory.
func ReadSamples(path string) ([]Sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dir := filepath.Dir(path)
	var out []Sample
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for n := 1; sc.Scan(); n++ {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var s Sample
		if err := json.Unmarshal(sc.Bytes(), &s); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, n, err)
		}
		if s.Box != nil && len(s.Box) != 4 {
			return nil, fmt.Errorf("%s:%d: box must have 4 values", path, n)
		}
		if !filepath.IsAbs(s.Image) {
			s.Image = filepath.Join(dir, s.Image)
		}
		out = append(out, s)
	}
	return out, sc.Err()
}

// BoxOf returns the sample's box, or the ink box of its page.
func (s Sample) BoxOf(page *image.Gray) Box {
	if s.Box == nil {
		return InkBox(page)
	}
	return Box{s.Box[0], s.Box[1], s.Box[2], s.Box[3]}
}

// ReadImage decodes a PNG or JPEG file to gray.
func ReadImage(path string) (*image.Gray, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return Gray(img), nil
}

// LoadLines reads labeled lines ready for training or evaluation: a shard
// (any extension but .jsonl), or a JSONL file of samples, cut with o. Texts
// are normalized (NormalizeText).
func LoadLines(path string, o CropOptions) ([]ShardLine, error) {
	if filepath.Ext(path) != ".jsonl" {
		r, err := OpenShard(path)
		if err != nil {
			return nil, err
		}
		defer r.Close()
		var out []ShardLine
		for {
			l, err := r.Next()
			if err == io.EOF {
				return out, nil
			}
			if err != nil {
				return nil, err
			}
			l.Text = NormalizeText(l.Text)
			out = append(out, l)
		}
	}
	samples, err := ReadSamples(path)
	if err != nil {
		return nil, err
	}
	out := make([]ShardLine, len(samples))
	pages := map[string]*image.Gray{}
	for i, s := range samples {
		page := pages[s.Image]
		if page == nil {
			if page, err = ReadImage(s.Image); err != nil {
				return nil, err
			}
			pages[s.Image] = page
		}
		c := CropLine(page, s.BoxOf(page), o)
		out[i] = ShardLine{Text: NormalizeText(s.Text), W: c.W, Rows: len(c.Ink) / c.W, Meta: s.Meta, Ink: c.Ink}
	}
	return out, nil
}
