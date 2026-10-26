package ocr

import (
	"encoding/json"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const pageDir = "../testdata/ocr-page"

type pageCase struct {
	Text  string    `json:"text"`
	Box   []float64 `json:"box"`
	Dark  bool      `json:"dark"`
	Width int       `json:"width"`
}

func loadPage(t testing.TB) (*image.Gray, []pageCase) {
	t.Helper()
	f, err := os.Open(filepath.Join(pageDir, "page.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(pageDir, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []pageCase
	if err := json.Unmarshal(b, &cases); err != nil {
		t.Fatal(err)
	}
	return Gray(img), cases
}

func (c pageCase) box() Box { return Box{c.Box[0], c.Box[1], c.Box[2], c.Box[3]} }

// TestCropMatchesOpenCV compares CropLine with Indra's crop_line, which
// resizes with OpenCV in uint8: up to a rounding, the same ink.
func TestCropMatchesOpenCV(t *testing.T) {
	page, cases := loadPage(t)
	want, err := os.ReadFile(filepath.Join(pageDir, "crops.u8"))
	if err != nil {
		t.Fatal(err)
	}
	o := DefaultCrop()
	rows := o.Height * o.Window
	off := 0
	for i, c := range cases {
		got := CropLine(page, c.box(), o)
		if got.W != c.Width || got.Dark != c.Dark {
			t.Fatalf("line %d: width %d dark %v, want %d %v", i, got.W, got.Dark, c.Width, c.Dark)
		}
		var sum, worst float64
		for j, v := range got.Ink {
			d := math.Abs(float64(v)*255 - float64(want[off+j]))
			sum += d
			worst = max(worst, d)
		}
		mean := sum / float64(len(got.Ink))
		t.Logf("line %d: mean |Δ| %.3f/255, max %.1f/255", i, mean, worst)
		if mean > 0.5 || worst > 8 {
			t.Errorf("line %d: crop differs from OpenCV's (mean %.3f, max %.1f)", i, mean, worst)
		}
		off += rows * c.Width
	}
}

func TestErrorRate(t *testing.T) {
	var e ErrorRate
	e.Add("le chat noir", "le chat noir")
	e.Add("le chut", "le chat")
	e.Add("", "abc")
	if e.Chars != 12+7+3 || e.CharEdits != 0+1+3 {
		t.Fatalf("chars %d edits %d", e.Chars, e.CharEdits)
	}
	if e.Words != 3+2+1 || e.WordEdits != 0+1+1 || e.Exact != 1 || e.Lines != 3 {
		t.Fatalf("%+v", e)
	}
	if got := levenshtein([]rune("kitten"), []rune("sitting")); got != 3 {
		t.Fatalf("levenshtein %d", got)
	}
}

func TestInkBox(t *testing.T) {
	page, cases := loadPage(t)
	for i, c := range cases {
		b := c.box()
		if c.Dark {
			continue // the banner's edge is ink against the page
		}
		sub := page.SubImage(image.Rect(int(b.X0)-25, int(b.Y0)-8, int(b.X1)+25, int(b.Y1)+8)).(*image.Gray)
		got := InkBox(sub)
		if math.Abs(got.X0-b.X0) > 4 || math.Abs(got.X1-b.X1) > 4 || math.Abs(got.Y0-b.Y0) > 4 || math.Abs(got.Y1-b.Y1) > 4 {
			t.Errorf("line %d: ink box %+v, want about %+v", i, got, b)
		}
	}
}
