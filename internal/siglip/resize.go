package siglip

import (
	"image"
	"image/color"
	"math"
)

// RGB returns the pixels of img as interleaved 8-bit RGB, [h, w, 3]. Alpha
// is dropped without compositing, as PIL's convert("RGB") does.
func RGB(img image.Image) (rgb []uint8, w, h int) {
	b := img.Bounds()
	w, h = b.Dx(), b.Dy()
	rgb = make([]uint8, w*h*3)
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
			rgb[i], rgb[i+1], rgb[i+2] = c.R, c.G, c.B
			i += 3
		}
	}
	return rgb, w, h
}

// Resize scales interleaved RGB bytes, [h, w, 3], to [outH, outW, 3] with
// the bilinear filter of PIL (Image.resize, BILINEAR), byte for byte: a
// triangle filter widened by the scale when downscaling (antialiasing),
// 22-bit fixed-point coefficients, a horizontal then a vertical pass, each
// rounded to bytes. That is what SigLIP's image processor feeds the model.
func Resize(rgb []uint8, w, h, outW, outH int) []uint8 {
	if w == outW && h == outH {
		return append([]uint8(nil), rgb...)
	}
	cur, cw := rgb, w
	if outW != w {
		cur = resizePass(cur, w, h, outW, true)
		cw = outW
	}
	if outH != h {
		cur = resizePass(cur, cw, h, outH, false)
	}
	return cur
}

const precisionBits = 32 - 8 - 2

// coeffs computes, for each output position, the first input position and
// the fixed-point weights of the inputs it reads (PIL's precompute_coeffs
// and normalize_coeffs_8bpc).
func coeffs(in, out int) (bounds []int, k [][]int32) {
	scale := float64(in) / float64(out)
	filterscale := max(scale, 1)
	support := 1.0 * filterscale // bilinear: support 1
	ss := 1 / filterscale
	bounds = make([]int, out)
	k = make([][]int32, out)
	for xx := 0; xx < out; xx++ {
		center := (float64(xx) + 0.5) * scale
		xmin := int(center - support + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > in {
			xmax = in
		}
		xmax -= xmin
		ws := make([]float64, xmax)
		var total float64
		for x := 0; x < xmax; x++ {
			t := math.Abs((float64(x+xmin) - center + 0.5) * ss)
			var v float64
			if t < 1 {
				v = 1 - t
			}
			ws[x] = v
			total += v
		}
		kk := make([]int32, xmax)
		for x, v := range ws {
			if total != 0 {
				v /= total
			}
			if v < 0 {
				kk[x] = int32(-0.5 + v*(1<<precisionBits))
			} else {
				kk[x] = int32(0.5 + v*(1<<precisionBits))
			}
		}
		bounds[xx], k[xx] = xmin, kk
	}
	return bounds, k
}

func clip8(v int32) uint8 {
	v >>= precisionBits
	switch {
	case v < 0:
		return 0
	case v > 255:
		return 255
	}
	return uint8(v)
}

// resizePass resamples one axis: horizontally (width w to out) or
// vertically (height h to out).
func resizePass(src []uint8, w, h, out int, horizontal bool) []uint8 {
	if horizontal {
		bounds, k := coeffs(w, out)
		dst := make([]uint8, out*h*3)
		for y := 0; y < h; y++ {
			row := src[y*w*3:]
			for xx := 0; xx < out; xx++ {
				for c := 0; c < 3; c++ {
					ss := int32(1 << (precisionBits - 1))
					for x, kv := range k[xx] {
						ss += int32(row[(bounds[xx]+x)*3+c]) * kv
					}
					dst[(y*out+xx)*3+c] = clip8(ss)
				}
			}
		}
		return dst
	}
	bounds, k := coeffs(h, out)
	dst := make([]uint8, w*out*3)
	for yy := 0; yy < out; yy++ {
		for x := 0; x < w; x++ {
			for c := 0; c < 3; c++ {
				ss := int32(1 << (precisionBits - 1))
				for y, kv := range k[yy] {
					ss += int32(src[((bounds[yy]+y)*w+x)*3+c]) * kv
				}
				dst[(yy*w+x)*3+c] = clip8(ss)
			}
		}
	}
	return dst
}

// Preprocess turns an image into the vision tower's input: RGB, resized to
// ImageSize × ImageSize with PIL's bilinear filter, normalized.
func (c Config) Preprocess(img image.Image) ([]float32, error) {
	return c.PreprocessInto(nil, img)
}

// PreprocessInto is Preprocess writing the pixels into dst when it is
// large enough.
func (c Config) PreprocessInto(dst []float32, img image.Image) ([]float32, error) {
	rgb, w, h := RGB(img)
	return PixelsInto(dst, Resize(rgb, w, h, c.ImageSize, c.ImageSize), c.ImageSize)
}
