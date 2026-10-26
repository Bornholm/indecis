package ocr

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

// Box is a line's ink box in page pixels: x0, y0 inclusive, x1, y1
// exclusive edges, fractional values allowed.
type Box struct {
	X0, Y0, X1, Y1 float64
}

// CropOptions sets how a line is cut from its page. The defaults are those
// of Indra OCR (rec_preproc.crop_line), which both training and reading
// must use: the text scale and the margins must match.
type CropOptions struct {
	Height   int     // line height in pixels after resizing (32)
	Window   int     // line heights in the window, the line in the middle (3)
	PadY     float64 // margin above and below the line, × its height (0.2)
	PadX     float64 // margin left and right, × its height (0.3)
	MaxWidth int     // widest resized line in pixels (4096)
	Multiple int     // the width is rounded up to a multiple of this (8)
}

// DefaultCrop returns the default crop options.
func DefaultCrop() CropOptions {
	return CropOptions{Height: 32, Window: 3, PadY: 0.2, PadX: 0.3, MaxWidth: 4096, Multiple: 8}
}

// Crop is a line cut from its page, as the recognizer reads it.
type Crop struct {
	Ink   []float32 // [Window·Height, W] row-major: 0 = paper, 1 = ink
	W     int
	Dark  bool    // light text on a dark ground, inverted
	X0    float64 // page x of the crop's left edge
	Scale float64 // page pixels per crop pixel
}

// PageX converts a crop x (resized pixels) to a page x.
func (c Crop) PageX(x float64) float64 { return c.X0 + x*c.Scale }

// Gray converts an image to 8-bit gray with the weights of OpenCV and PIL
// (0.299 R + 0.587 G + 0.114 B).
func Gray(img image.Image) *image.Gray {
	if g, ok := img.(*image.Gray); ok {
		return g
	}
	b := img.Bounds()
	g := image.NewGray(b)
	if _, ok := img.(*image.YCbCr); ok {
		// Y is already the luma.
		draw.Draw(g, b, img, b.Min, draw.Src)
		return g
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, gg, bb, a := img.At(x, y).RGBA()
			// Transparent pixels over white paper.
			w := 0xffff - a
			lum := (299*(r+w) + 587*(gg+w) + 114*(bb+w) + 500) / 1000
			g.SetGray(x, y, color.Gray{Y: uint8(min(lum>>8, 255))})
		}
	}
	return g
}

// IsDark reports whether a line is light text on a dark ground (a banner,
// a dark table head): the ground, the median of the box, lies nearer the
// dark end than the text does.
func IsDark(page *image.Gray, box Box) bool {
	b := page.Bounds()
	x0, y0 := max(b.Min.X, int(box.X0)), max(b.Min.Y, int(box.Y0))
	x1, y1 := min(b.Max.X, int(box.X1)+1), min(b.Max.Y, int(box.Y1)+1)
	if x1 <= x0 || y1 <= y0 {
		return false
	}
	var hist [256]int
	for y := y0; y < y1; y++ {
		row := page.Pix[page.PixOffset(x0, y):page.PixOffset(x1, y)]
		for _, v := range row {
			hist[v]++
		}
	}
	n := (x1 - x0) * (y1 - y0)
	lo, med, hi := percentile(&hist, n, 0.02), percentile(&hist, n, 0.5), percentile(&hist, n, 0.98)
	return hi-med > 2*(med-lo)+8
}

// percentile is numpy's linear percentile of the values counted in hist.
func percentile(hist *[256]int, n int, q float64) float64 {
	x := q * float64(n-1)
	i := int(math.Floor(x))
	g := x - float64(i)
	at := func(k int) float64 {
		c := 0
		for v, h := range hist {
			c += h
			if c > k {
				return float64(v)
			}
		}
		return 255
	}
	lo, hi := at(i), at(min(i+1, n-1))
	if g >= 0.5 {
		return hi - (hi-lo)*(1-g)
	}
	return lo + (hi-lo)*g
}

// CropLine cuts the line in box from page: the box widened by the margins
// and by the window, paper outside the page, resized to Window·Height
// pixels high keeping the aspect ratio (area averaging when shrinking,
// bilinear when enlarging).
func CropLine(page *image.Gray, box Box, o CropOptions) Crop {
	return cropLine(page, box, o, IsDark(page, box))
}

func cropLine(page *image.Gray, box Box, o CropOptions, dark bool) Crop {
	padY := (float64(o.Window)*(1+2*o.PadY) - 1) / 2
	height := o.Height * o.Window

	h := max(1, box.Y1-box.Y0)
	ix0 := int(math.Floor(box.X0 - o.PadX*h))
	iy0 := int(math.Floor(box.Y0 - padY*h))
	ix1 := int(math.Ceil(box.X1 + o.PadX*h))
	iy1 := int(math.Ceil(box.Y1 + padY*h))
	cw, ch := max(1, ix1-ix0), max(1, iy1-iy0)

	paper := uint8(255)
	if dark {
		paper = 0
	}
	crop := make([]uint8, cw*ch)
	b := page.Bounds()
	for y := range ch {
		row := crop[y*cw : (y+1)*cw]
		py := iy0 + y
		for x := range row {
			px := ix0 + x
			if px >= b.Min.X && px < b.Max.X && py >= b.Min.Y && py < b.Max.Y {
				row[x] = page.Pix[page.PixOffset(px, py)]
			} else {
				row[x] = paper
			}
		}
	}

	outW := int(math.Round(float64(cw) * float64(height) / float64(ch)))
	outW = max(o.Multiple, min(o.MaxWidth, outW))
	var resized []float32
	if ch > height {
		resized = resizeArea(crop, cw, ch, outW, height)
	} else {
		resized = resizeLinear(crop, cw, ch, outW, height)
	}
	for i, v := range resized {
		if dark {
			resized[i] = v / 255
		} else {
			resized[i] = 1 - v/255
		}
	}
	return Crop{Ink: resized, W: outW, Dark: dark, X0: float64(ix0), Scale: float64(cw) / float64(outW)}
}

// resizeArea shrinks src [sh, sw] to [dh, dw] by area averaging: each
// output pixel is the mean of the source over its footprint, partial
// pixels weighted by their covered fraction (OpenCV's INTER_AREA).
func resizeArea(src []uint8, sw, sh, dw, dh int) []float32 {
	wx := areaWeights(sw, dw)
	wy := areaWeights(sh, dh)
	tmp := make([]float32, sh*dw) // rows resized horizontally
	for y := range sh {
		row := src[y*sw : (y+1)*sw]
		out := tmp[y*dw : (y+1)*dw]
		for _, t := range wx {
			out[t.dst] += t.w * float32(row[t.src])
		}
	}
	out := make([]float32, dh*dw)
	for _, t := range wy {
		axpy(out[t.dst*dw:(t.dst+1)*dw], tmp[t.src*dw:(t.src+1)*dw], t.w)
	}
	return out
}

type tap struct {
	src, dst int
	w        float32
}

// areaWeights lists the (source, destination, weight) taps of an area
// resize along one axis, n source pixels to m.
func areaWeights(n, m int) []tap {
	scale := float64(n) / float64(m)
	var taps []tap
	for d := range m {
		f0, f1 := float64(d)*scale, float64(d+1)*scale
		for s := int(math.Floor(f0)); s < int(math.Ceil(f1)) && s < n; s++ {
			cover := min(f1, float64(s+1)) - max(f0, float64(s))
			if cover > 1e-9 {
				taps = append(taps, tap{src: s, dst: d, w: float32(cover / scale)})
			}
		}
	}
	return taps
}

// resizeLinear enlarges src [sh, sw] to [dh, dw] bilinearly, pixel centers
// aligned (OpenCV's INTER_LINEAR, PIL's BILINEAR when enlarging).
func resizeLinear(src []uint8, sw, sh, dw, dh int) []float32 {
	type lerp struct {
		i0, i1 int
		f      float32
	}
	axis := func(n, m int) []lerp {
		out := make([]lerp, m)
		scale := float64(n) / float64(m)
		for d := range out {
			x := (float64(d)+0.5)*scale - 0.5
			x = max(0, x)
			i0 := min(int(x), n-1)
			out[d] = lerp{i0, min(i0+1, n-1), float32(x - float64(i0))}
		}
		return out
	}
	lx, ly := axis(sw, dw), axis(sh, dh)
	out := make([]float32, dh*dw)
	for y, a := range ly {
		r0, r1 := src[a.i0*sw:(a.i0+1)*sw], src[a.i1*sw:(a.i1+1)*sw]
		o := out[y*dw : (y+1)*dw]
		for x, b := range lx {
			top := float32(r0[b.i0]) + (float32(r0[b.i1])-float32(r0[b.i0]))*b.f
			bot := float32(r1[b.i0]) + (float32(r1[b.i1])-float32(r1[b.i0]))*b.f
			o[x] = top + (bot-top)*a.f
		}
	}
	return out
}

func axpy(o, x []float32, a float32) {
	for i, v := range x {
		o[i] += a * v
	}
}

// InkBox returns the bounding box of the dark pixels of an image holding a
// single line (a line image from a dataset), or the whole image when it has
// no ink. A pixel is ink when darker than the midpoint between the image's
// 2nd and 98th percentiles.
func InkBox(page *image.Gray) Box {
	b := page.Bounds()
	var hist [256]int
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for _, v := range page.Pix[page.PixOffset(b.Min.X, y):page.PixOffset(b.Max.X, y)] {
			hist[v]++
		}
	}
	n := b.Dx() * b.Dy()
	full := Box{float64(b.Min.X), float64(b.Min.Y), float64(b.Max.X), float64(b.Max.Y)}
	if n == 0 {
		return full
	}
	lo, hi := percentile(&hist, n, 0.02), percentile(&hist, n, 0.98)
	if hi-lo < 32 {
		return full
	}
	dark := IsDark(page, full)
	thr := uint8((lo + hi) / 2)
	x0, y0, x1, y1 := b.Max.X, b.Max.Y, b.Min.X, b.Min.Y
	for y := b.Min.Y; y < b.Max.Y; y++ {
		row := page.Pix[page.PixOffset(b.Min.X, y):page.PixOffset(b.Max.X, y)]
		for i, v := range row {
			if (v < thr) != dark {
				x := b.Min.X + i
				x0, x1 = min(x0, x), max(x1, x+1)
				y0, y1 = min(y0, y), max(y1, y+1)
			}
		}
	}
	if x1 <= x0 {
		return full
	}
	return Box{float64(x0), float64(y0), float64(x1), float64(y1)}
}
