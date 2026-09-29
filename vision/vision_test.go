package vision

import (
	"context"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/bornholm/indecis"
)

func load(t *testing.T) *Model {
	t.Helper()
	dir := os.Getenv("INDECIS_SIGLIP2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256")
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("SigLIP 2 model absent (%s): set INDECIS_SIGLIP2_DIR", dir)
	}
	m, err := Load(dir, WithInt8())
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
