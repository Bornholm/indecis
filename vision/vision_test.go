package vision

import (
	"context"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/bornholm/indecis"
)

func load(t *testing.T, opts ...Option) *Model {
	t.Helper()
	dir := os.Getenv("INDECIS_SIGLIP2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256")
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("SigLIP 2 model absent (%s): set INDECIS_SIGLIP2_DIR", dir)
	}
	m, err := Load(dir, append([]Option{WithInt8()}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func open(t *testing.T, name string) image.Image {
	t.Helper()
	f, err := os.Open("../testdata/siglip2/" + name + ".png")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func TestDecide(t *testing.T) {
	m := load(t)
	ctx := context.Background()
	shapes, noise := open(t, "shapes"), open(t, "noise")

	p, err := m.Match(shapes, "a red circle on a blue background")
	if err != nil {
		t.Fatal(err)
	}
	if p < 0.9 {
		t.Errorf("Match(shapes, red circle) = %.3f, want > 0.9", p)
	}
	if p, _ := m.Match(shapes, "a photo of a cat"); p > 0.1 {
		t.Errorf("Match(shapes, cat) = %.3f, want < 0.1", p)
	}

	cands := []indecis.Candidate{
		{Name: "shapes", Description: "un cercle rouge sur fond bleu"},
		{Name: "noise", Description: "random colored noise"},
		{Name: "cat", Description: "a photo of a cat"},
	}
	for img, want := range map[image.Image]string{shapes: "shapes", noise: "noise"} {
		a, err := m.ChooseNearest(ctx, cands, img)
		if err != nil {
			t.Fatal(err)
		}
		if a.Choice != want {
			t.Errorf("ChooseNearest = %s (%v), want %s", a.Choice, a.Probs, want)
		}
	}

	// An option's examples count: they outweigh a misleading description.
	withExamples := []indecis.Candidate{
		{Name: "shapes", Description: "a photo of a cat", Examples: []string{"a red circle on a blue background", "un cercle rouge sur fond bleu", "a red disk"}},
		{Name: "noise", Description: "random colored noise"},
	}
	if a, err := m.ChooseNearest(ctx, withExamples, shapes); err != nil || a.Choice != "shapes" {
		t.Errorf("ChooseNearest with examples = %s (%v, %v), want shapes", a.Choice, a.Probs, err)
	}

	qs := []indecis.OpenQuestion{
		indecis.OpenNoul("circle", "a red circle"),
		{Name: "content", Kind: indecis.Choice, Options: cands},
	}
	d, err := m.DecideOpen(ctx, qs, shapes, noise)
	if err != nil {
		t.Fatal(err)
	}
	if d[0]["circle"].P < 0.5 || d[1]["circle"].P > 0.5 {
		t.Errorf("circle: P = %.3f on shapes, %.3f on noise", d[0]["circle"].P, d[1]["circle"].P)
	}
	if d[0]["content"].Choice != "shapes" || d[1]["content"].Choice != "noise" {
		t.Errorf("content: %s, %s", d[0]["content"].Choice, d[1]["content"].Choice)
	}
}

// A head trained on patch features, saved and loaded back, answers its
// learned question in the same pass as an open one.
func TestTrainedModel(t *testing.T) {
	m := load(t)
	shapes, noise := open(t, "shapes"), open(t, "noise")
	var examples []HeadExample
	for _, c := range []struct {
		img   image.Image
		label string
	}{{shapes, "shapes"}, {noise, "noise"}} {
		p, err := m.Patches(c.img)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 20; i++ {
			examples = append(examples, HeadExample{Patches: ToBF16(p), Labels: map[string]any{"kind": c.label}})
		}
	}
	T, H := m.PatchShape()
	h, err := NewHead(indecis.Schema{indecis.NewChoice("kind", "", "shapes", "noise")}, T, H, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Fit(examples, DefaultHeadTrainOptions()); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	home, _ := os.UserHomeDir()
	backbone := os.Getenv("INDECIS_SIGLIP2_DIR")
	if backbone == "" {
		backbone = filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256")
	}
	if err := SaveHead(dir, backbone, h); err != nil {
		t.Fatal(err)
	}
	if !IsModel(dir) {
		t.Fatal("IsModel(trained dir) = false")
	}

	// A head reading a layer the encoder does not have is refused at load.
	bad := t.TempDir()
	if err := SaveHead(bad, backbone, &Head{Schema: h.Schema, T: h.T, H: h.H, K: h.K, Layer: 99, W1: h.W1, B1: h.B1, W2: h.W2, B2: h.B2}); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(bad); err == nil || !strings.Contains(err.Error(), "layer 99") {
		t.Errorf("layer 99: %v", err)
	}
	trained, err := Load(dir, WithInt8())
	if err != nil {
		t.Fatal(err)
	}
	d, err := trained.Decide(context.Background(), noise, []string{"kind"}, []indecis.OpenQuestion{indecis.OpenNoul("circle", "a red circle")})
	if err != nil {
		t.Fatal(err)
	}
	if d["kind"].Choice != "noise" {
		t.Errorf("kind = %s (%v), want noise", d["kind"].Choice, d["kind"].Probs)
	}
	if d["circle"].P > 0.5 {
		t.Errorf("circle on noise: P = %.3f", d["circle"].P)
	}
}

// Simultaneous requests for a text out of the cache share one pass of the
// text tower. The cache is off: only that sharing gives them one slice.
func TestEmbedTextShared(t *testing.T) {
	m := load(t, WithEmbedCache(0))
	if _, err := m.EmbedText("warm-up"); err != nil { // loads the text tower
		t.Fatal(err)
	}
	start := make(chan struct{})
	got := make([][]float32, 4)
	var wg sync.WaitGroup
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			got[i], _ = m.EmbedText("a photo of a dog")
		}()
	}
	close(start)
	wg.Wait()
	for i, e := range got {
		if len(e) == 0 || &e[0] != &got[0][0] {
			t.Fatalf("call %d computed its own embedding", i)
		}
	}
	later, err := m.EmbedText("a photo of a dog")
	if err != nil || &later[0] == &got[0][0] {
		t.Fatalf("a later call, cache off, reused the shared result (err %v)", err)
	}
}
