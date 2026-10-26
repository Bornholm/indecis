package main

import (
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"

	"github.com/bornholm/indecis/ocr"
)

// TestFaceMetrics: x/image reads the metrics of every font loaded.
func TestFaceMetrics(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".cache/indecis/fonts")
	fonts, err := loadFonts([]string{dir}, []rune(ocr.DefaultCharset))
	if err != nil {
		t.Skip(err)
	}
	bad := 0
	for _, f := range fonts {
		for size := 12; size <= 56; size++ {
			face, err := opentype.NewFace(f.f, &opentype.FaceOptions{Size: float64(size), DPI: 72, Hinting: font.HintingNone})
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := metrics(face); !ok {
				bad++
				t.Logf("%s at %d", f.name, size)
				break
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d/%d fonts fail", bad, len(fonts))
	}
}
