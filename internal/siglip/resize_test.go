package siglip

import (
	"bytes"
	"image"
	"math/rand"
	"testing"
)

// resizeImage converts and resamples row by row: same bytes as the whole
// image converted, then resized.
func TestResizeImageMatchesResize(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cases := []struct {
		name string
		img  image.Image
		size int
	}{
		{"downscale", randomNRGBA(rng, image.Rect(0, 0, 613, 401)), 256},
		{"upscale", randomNRGBA(rng, image.Rect(0, 0, 97, 130)), 256},
		{"same width", randomNRGBA(rng, image.Rect(0, 0, 256, 300)), 256},
		{"same size", randomNRGBA(rng, image.Rect(0, 0, 256, 256)), 256},
		{"offset bounds", randomNRGBA(rng, image.Rect(0, 0, 500, 500)).SubImage(image.Rect(37, 81, 420, 333)), 256},
		{"YCbCr", randomYCbCr(rng, image.Rect(0, 0, 341, 257)), 256},
		{"tall and narrow", randomNRGBA(rng, image.Rect(0, 0, 3, 5000)), 256},
		{"wide and short", randomNRGBA(rng, image.Rect(0, 0, 5000, 3)), 256},
		{"same height", randomNRGBA(rng, image.Rect(0, 0, 700, 256)), 256},
		{"upscale, odd target", randomNRGBA(rng, image.Rect(0, 0, 13, 7)), 31},
	}
	for _, c := range cases {
		rgb, w, h := RGB(c.img)
		want := Resize(rgb, w, h, c.size, c.size)
		if got := resizeImage(c.img, c.size, c.size); !bytes.Equal(got, want) {
			t.Errorf("%s: resizeImage differs from Resize(RGB(img))", c.name)
		}
	}
}

// The direct reads of rgbRow give the bytes of color.NRGBAModel.
func TestRGBRowMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	r := image.Rect(3, 5, 131, 77)
	rgba := image.NewRGBA(r)
	rng.Read(rgba.Pix)
	for i := 0; i < len(rgba.Pix); i += 4 { // premultiplied: color <= alpha
		a := rgba.Pix[i+3]
		for c := 0; c < 3; c++ {
			rgba.Pix[i+c] = min(rgba.Pix[i+c], a)
		}
	}
	gray := image.NewGray(r)
	rng.Read(gray.Pix)
	for _, img := range []image.Image{randomNRGBA(rng, r), rgba, randomYCbCr(rng, r), gray} {
		got := make([]uint8, r.Dx()*3)
		want := make([]uint8, r.Dx()*3)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			rgbRow(got, img, y)
			rgbRowGeneric(want, img, y)
			if !bytes.Equal(got, want) {
				t.Fatalf("%T, row %d: direct read differs from color.NRGBAModel", img, y)
			}
		}
	}
}

func randomNRGBA(rng *rand.Rand, r image.Rectangle) *image.NRGBA {
	img := image.NewNRGBA(r)
	rng.Read(img.Pix)
	return img
}

func randomYCbCr(rng *rand.Rand, r image.Rectangle) *image.YCbCr {
	img := image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
	rng.Read(img.Y)
	rng.Read(img.Cb)
	rng.Read(img.Cr)
	return img
}

func TestPreprocessBoundsSides(t *testing.T) {
	c := Config{ImageSize: 256}
	for _, r := range []image.Rectangle{image.Rect(0, 0, 1, MaxImageSide+1), image.Rect(0, 0, MaxImageSide+1, 1), image.Rect(0, 0, 0, 5)} {
		if _, err := c.Preprocess(image.NewGray(r)); err == nil {
			t.Errorf("%v: accepted", r)
		}
	}
}
