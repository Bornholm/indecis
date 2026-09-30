package siglip

import (
	"encoding/json"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/bornholm/indecis/tokenizer"
)

func modelDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("INDECIS_SIGLIP2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256")
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("SigLIP 2 model absent (%s): set INDECIS_SIGLIP2_DIR", dir)
	}
	return dir
}

var (
	models   = map[bool]*Model{}
	modelsMu sync.Mutex
)

func load(t testing.TB, quantized bool) *Model {
	t.Helper()
	dir := modelDir(t)
	modelsMu.Lock()
	defer modelsMu.Unlock()
	if m, ok := models[quantized]; ok {
		return m
	}
	m, err := Load(dir, quantized)
	if err != nil {
		t.Fatal(err)
	}
	models[quantized] = m
	return m
}

type cases struct {
	Images []struct {
		Name      string    `json:"name"`
		Embedding []float32 `json:"embedding"`
	} `json:"images"`
	Texts []struct {
		Text      string    `json:"text"`
		IDs       []int32   `json:"ids"`
		Embedding []float32 `json:"embedding"`
	} `json:"texts"`
	Logits [][]float32 `json:"logits_per_image"`
}

func readCases(t *testing.T) cases {
	t.Helper()
	b, err := os.ReadFile("../../testdata/siglip2/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var c cases
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

func imageEmbedding(t *testing.T, m *Model, name string) []float32 {
	t.Helper()
	rgb, err := os.ReadFile("../../testdata/siglip2/" + name + "_resized.u8")
	if err != nil {
		t.Fatal(err)
	}
	px, err := Pixels(rgb, m.Cfg.ImageSize)
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Vision.Embed(px)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// maxRelDiff is max |a-b| / max |b|.
func maxRelDiff(a, b []float32) float64 {
	var d, n float64
	for i := range b {
		d = max(d, math.Abs(float64(a[i]-b[i])))
		n = max(n, math.Abs(float64(b[i])))
	}
	return d / n
}

func cosine(a, b []float32) float32 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return float32(d / math.Sqrt(na*nb))
}

// Parity with transformers, float32: embeddings and logits.
func TestParity(t *testing.T) {
	m := load(t, false)
	c := readCases(t)
	imgs := make([][]float32, len(c.Images))
	for i, im := range c.Images {
		imgs[i] = imageEmbedding(t, m, im.Name)
		t.Logf("image %s: max relative difference %.1e", im.Name, maxRelDiff(imgs[i], im.Embedding))
		if d := maxRelDiff(imgs[i], im.Embedding); d > 1e-4 {
			t.Errorf("image %s: max relative difference %.2e", im.Name, d)
		}
	}
	txts := make([][]float32, len(c.Texts))
	for i, tx := range c.Texts {
		e, err := mustText(t, m).Embed(tx.IDs)
		if err != nil {
			t.Fatal(err)
		}
		txts[i] = e
		t.Logf("text %d: max relative difference %.1e", i, maxRelDiff(e, tx.Embedding))
		if d := maxRelDiff(e, tx.Embedding); d > 1e-4 {
			t.Errorf("text %q: max relative difference %.2e", tx.Text, d)
		}
	}
	for i := range imgs {
		for j := range txts {
			got, want := m.Logit(cosine(imgs[i], txts[j])), c.Logits[i][j]
			if math.Abs(float64(got-want)) > 1e-3 {
				t.Errorf("logit[%d][%d] = %.5f, want %.5f", i, j, got, want)
			}
		}
	}
}

// int8 keeps the decisions: logits within 0.9 of the reference, and the
// same best text for each image. Short captions stay within 0.75; the
// 64-token list caption reaches 0.85. On 300 Imagenette images, the mean gap
// is 0.21 and zero-shot accuracy is the same as in float32 (99.7%).
func TestInt8(t *testing.T) {
	m := load(t, true)
	c := readCases(t)
	txts := make([][]float32, len(c.Texts))
	for j, tx := range c.Texts {
		e, err := mustText(t, m).Embed(tx.IDs)
		if err != nil {
			t.Fatal(err)
		}
		txts[j] = e
	}
	for i, im := range c.Images {
		img := imageEmbedding(t, m, im.Name)
		best, bestRef := 0, 0
		var worst float64
		for j := range txts {
			got, want := m.Logit(cosine(img, txts[j])), c.Logits[i][j]
			worst = max(worst, math.Abs(float64(got-want)))
			if got > m.Logit(cosine(img, txts[best])) {
				best = j
			}
			if want > c.Logits[i][bestRef] {
				bestRef = j
			}
		}
		t.Logf("image %s: max logit difference %.3f", im.Name, worst)
		if worst > 0.9 {
			t.Errorf("image %s: logits differ by %.3f", im.Name, worst)
		}
		if best != bestRef {
			t.Errorf("image %s: best text %d, want %d", im.Name, best, bestRef)
		}
	}
}

func benchVision(b *testing.B, quantized bool) {
	m := load(b, quantized)
	px := make([]float32, 3*m.Cfg.ImageSize*m.Cfg.ImageSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Vision.Embed(px); err != nil {
			b.Fatal(err)
		}
	}
}

func benchText(b *testing.B, quantized bool) {
	m := load(b, quantized)
	tt := mustText(b, m)
	ids := make([]int32, m.Cfg.TextLen)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tt.Embed(ids); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVisionFloat32(b *testing.B) { benchVision(b, false) }
func BenchmarkVisionInt8(b *testing.B)    { benchVision(b, true) }
func BenchmarkTextFloat32(b *testing.B)   { benchText(b, false) }
func BenchmarkTextInt8(b *testing.B)      { benchText(b, true) }

// Resize matches PIL's bilinear resize byte for byte: downscaling one
// axis and upscaling the other (shapes, 320×240), upscaling both (noise,
// 131×97).
func TestResizeMatchesPIL(t *testing.T) {
	for _, name := range []string{"shapes", "noise"} {
		f, err := os.Open("../../testdata/siglip2/" + name + ".png")
		if err != nil {
			t.Fatal(err)
		}
		img, err := png.Decode(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile("../../testdata/siglip2/" + name + "_resized.u8")
		if err != nil {
			t.Fatal(err)
		}
		rgb, w, h := RGB(img)
		got := Resize(rgb, w, h, 256, 256)
		diff := 0
		for i := range want {
			if got[i] != want[i] {
				diff++
			}
		}
		if diff > 0 {
			t.Errorf("%s: %d of %d bytes differ from PIL", name, diff, len(want))
		}
	}
}

// Stopping at a layer gives the same features as capturing that layer in
// a full pass, and the full pass gives the same pooled embedding.
func TestForwardLayers(t *testing.T) {
	m := load(t, true)
	rgb, err := os.ReadFile("../../testdata/siglip2/shapes_resized.u8")
	if err != nil {
		t.Fatal(err)
	}
	px, _ := Pixels(rgb, m.Cfg.ImageSize)
	stopped, pooled0, err := m.Vision.Forward(px, 8, false)
	if err != nil || pooled0 != nil {
		t.Fatalf("Forward(8, no pool): %v, pooled %v", err, pooled0 != nil)
	}
	captured, pooled, err := m.Vision.Forward(px, 8, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := range stopped {
		if stopped[i] != captured[i] {
			t.Fatalf("layer 8: stopped and captured differ at %d", i)
		}
	}
	want, err := m.Vision.Embed(px)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if pooled[i] != want[i] {
			t.Fatalf("pooled embedding differs at %d", i)
		}
	}
}

func mustText(t testing.TB, m *Model) *Text {
	t.Helper()
	tt, err := m.Text()
	if err != nil {
		t.Fatal(err)
	}
	return tt
}

// Tokenizing then framing with Pad gives the ids of transformers' processor
// (padding and truncation to 64), including for a text beyond 64 tokens.
func TestPadMatchesProcessor(t *testing.T) {
	m := load(t, true)
	tok, err := tokenizer.Load(filepath.Join(modelDir(t), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	tt := mustText(t, m)
	long := false
	for _, tx := range readCases(t).Texts {
		ids := tok.Encode(tx.Text)
		long = long || len(ids) > m.Cfg.TextLen
		if got := tt.Pad(ids, tok.EosID(), tok.PadID()); !slices.Equal(got, tx.IDs) {
			t.Errorf("%q:\n got  %v\n want %v", tx.Text, got, tx.IDs)
		}
	}
	if !long {
		t.Error("no fixture text beyond the tower's length: truncation untested")
	}
}
