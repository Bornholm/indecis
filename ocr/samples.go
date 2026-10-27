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

// ReadSamples reads a JSONL file of samples; relative image paths are
// resolved against the JSONL file's directory.
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
// (any extension but .jsonl), whose lines must have been cut at o's height
// and window, or a JSONL file of samples, cut with o. Texts are normalized
// (NormalizeText).
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
			if l.Rows != o.Height*o.Window {
				return nil, fmt.Errorf("%s: lines %d pixels high, the model reads %d", path, l.Rows, o.Height*o.Window)
			}
			l.Text = NormalizeText(l.Text)
			out = append(out, l)
		}
	}
	var out []ShardLine
	err := EachLine(path, o, func(l ShardLine) error {
		out = append(out, l)
		return nil
	})
	return out, err
}

// EachLine cuts the lines of a JSONL file of samples one by one, holding a
// single page in memory: consecutive samples of the same image share it.
func EachLine(path string, o CropOptions, fn func(ShardLine) error) error {
	samples, err := ReadSamples(path)
	if err != nil {
		return err
	}
	var page *image.Gray
	var pagePath string
	for _, s := range samples {
		if s.Image != pagePath {
			if page, err = ReadImage(s.Image); err != nil {
				return err
			}
			pagePath = s.Image
		}
		c := CropLine(page, s.BoxOf(page), o)
		if err := fn(ShardLine{Text: NormalizeText(s.Text), W: c.W, Rows: len(c.Ink) / c.W, Meta: s.Meta, Ink: c.Ink}); err != nil {
			return err
		}
	}
	return nil
}

// PageSample is a labeled page (one JSONL row): its image and the ink box
// and text of each of its lines, every line, readable or not. It trains and
// measures line detection.
type PageSample struct {
	Image string       `json:"image"` // relative to the JSONL file
	Boxes [][4]float64 `json:"boxes"`
	Texts []string     `json:"texts"`
	// Ignore are regions whose text is not labeled (a figure's, when the
	// lines come from a PDF's text layer): the detector learns nothing
	// there.
	Ignore [][4]float64      `json:"ignore,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

// ReadPageSamples reads a JSONL file of pages; relative image paths are
// resolved against the JSONL file's directory.
func ReadPageSamples(path string) ([]PageSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dir := filepath.Dir(path)
	var out []PageSample
	dec := json.NewDecoder(f)
	for {
		var p PageSample
		if err := dec.Decode(&p); err == io.EOF {
			return out, nil
		} else if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if len(p.Texts) != len(p.Boxes) {
			return nil, fmt.Errorf("%s: %s: %d texts for %d boxes", path, p.Image, len(p.Texts), len(p.Boxes))
		}
		if !filepath.IsAbs(p.Image) {
			p.Image = filepath.Join(dir, p.Image)
		}
		out = append(out, p)
	}
}
