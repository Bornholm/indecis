package main

import (
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/font/sfnt"
)

// fontFile is a parsed font and the characters it draws.
type fontFile struct {
	name   string
	family string // the file name up to its style: "NotoSans" for NotoSans-BoldItalic.ttf
	f      *opentype.Font
	covers map[rune]bool
	broken atomic.Bool // x/image failed on it: never drawn again
}

// loadFonts parses the TrueType and OpenType fonts under dirs, keeping
// those that draw at least the letters and digits of ASCII.
func loadFonts(dirs []string, charset []rune) ([]*fontFile, error) {
	var out []*fontFile
	var buf sfnt.Buffer
	for _, dir := range dirs {
		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			ext := strings.ToLower(filepath.Ext(path))
			// Variable fonts ([wght] in their name) crash x/image's metrics:
			// instantiate them first (fonttools varLib.instancer).
			if d.IsDir() || (ext != ".ttf" && ext != ".otf") || strings.Contains(path, "[") || strings.Contains(path, "%5B") {
				return nil
			}
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			f, err := opentype.Parse(b)
			if err != nil {
				return nil // unsupported (a variable CFF2 font, a broken file)
			}
			name := filepath.Base(path)
			family := name[:len(name)-len(filepath.Ext(name))]
			if i := strings.IndexAny(family, "-["); i > 0 {
				family = family[:i]
			}
			ff := &fontFile{name: name, family: family, f: f, covers: map[rune]bool{}}
			for _, r := range append([]rune{'ſ'}, charset...) {
				if i, err := f.GlyphIndex(&buf, r); err == nil && i != 0 {
					ff.covers[r] = true
				}
			}
			if !usable(f) {
				return nil
			}
			ff.covers[' '] = true
			for _, r := range "AZaz09.,'" {
				if !ff.covers[r] {
					return nil
				}
			}
			out = append(out, ff)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no usable font under %v", dirs)
	}
	return out, nil
}

// fontSet draws fonts by family first, then a style: a family of 72 styles
// is not 72 times more likely than one with a single file.
type fontSet struct {
	families [][]*fontFile
}

func newFontSet(fonts []*fontFile) *fontSet {
	idx := map[string]int{}
	s := &fontSet{}
	for _, f := range fonts {
		i, ok := idx[f.family]
		if !ok {
			i = len(s.families)
			idx[f.family] = i
			s.families = append(s.families, nil)
		}
		s.families[i] = append(s.families[i], f)
	}
	return s
}

func (s *fontSet) pick(r *rand.Rand) *fontFile {
	for {
		if f := pick(r, pick(r, s.families)); !f.broken.Load() {
			return f
		}
	}
}

func (f *fontFile) draws(s string) bool {
	for _, r := range s {
		if !f.covers[r] {
			return false
		}
	}
	return true
}

// faces caches font faces per (font, size): creating one costs more than
// drawing a line.
type faces map[faceKey]font.Face

type faceKey struct {
	f    *fontFile
	size int
}

func (c faces) get(f *fontFile, size int) (font.Face, error) {
	k := faceKey{f, size}
	if face, ok := c[k]; ok {
		return face, nil
	}
	face, err := opentype.NewFace(f.f, &opentype.FaceOptions{Size: float64(size), DPI: 72, Hinting: font.HintingNone})
	if err != nil {
		return nil, err
	}
	if len(c) > 400 {
		clear(c)
	}
	c[k] = face
	return face, nil
}

// usable reports whether x/image can draw with f: sfnt panics on the
// metrics of some fonts (a missing OS/2 table).
func usable(f *opentype.Font) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	face, err := opentype.NewFace(f, &opentype.FaceOptions{Size: 20, DPI: 72})
	if err != nil {
		return false
	}
	m := face.Metrics()
	font.MeasureString(face, "Ag")
	return m.Ascent > 0
}
