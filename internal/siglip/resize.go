package siglip

import (
	"fmt"
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
	for y := 0; y < h; y++ {
		rgbRow(rgb[y*w*3:(y+1)*w*3], img, b.Min.Y+y)
	}
	return rgb, w, h
}

// rgbRow writes row y of img to dst as interleaved RGB. The image types
// the decoders return are read directly, with the same bytes as
// color.NRGBAModel: img.At would allocate for each pixel.
func rgbRow(dst []uint8, img image.Image, y int) {
	b := img.Bounds()
	switch m := img.(type) {
	case *image.NRGBA:
		src := m.Pix[m.PixOffset(b.Min.X, y):]
		for i := 0; i < b.Dx(); i++ {
			dst[3*i], dst[3*i+1], dst[3*i+2] = src[4*i], src[4*i+1], src[4*i+2]
		}
	case *image.RGBA:
		for i, x := 0, b.Min.X; x < b.Max.X; i, x = i+3, x+1 {
			dst[i], dst[i+1], dst[i+2] = unpremultiply(m.RGBAAt(x, y).RGBA())
		}
	case *image.YCbCr:
		for i, x := 0, b.Min.X; x < b.Max.X; i, x = i+3, x+1 {
			dst[i], dst[i+1], dst[i+2] = unpremultiply(m.YCbCrAt(x, y).RGBA())
		}
	case *image.Gray:
		for i, x := 0, b.Min.X; x < b.Max.X; i, x = i+3, x+1 {
			v := m.GrayAt(x, y).Y
			dst[i], dst[i+1], dst[i+2] = v, v, v
		}
	default:
		rgbRowGeneric(dst, img, y)
	}
}

func rgbRowGeneric(dst []uint8, img image.Image, y int) {
	b := img.Bounds()
	for i, x := 0, b.Min.X; x < b.Max.X; i, x = i+3, x+1 {
		c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
		dst[i], dst[i+1], dst[i+2] = c.R, c.G, c.B
	}
}

// unpremultiply is color.NRGBAModel's conversion, without the alpha.
func unpremultiply(r, g, b, a uint32) (uint8, uint8, uint8) {
	switch a {
	case 0xffff:
		return uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)
	case 0:
		return 0, 0, 0
	}
	return uint8(r * 0xffff / a >> 8), uint8(g * 0xffff / a >> 8), uint8(b * 0xffff / a >> 8)
}

// resizeImage is Resize(RGB(img)), byte for byte, in bounded memory: each
// source row is converted, resampled horizontally, then added at once to
// the output rows whose vertical window reads it. The integer sums do not
// depend on their order, so the result is the same as PIL's two passes;
// memory stays at one source row plus the output accumulators, whatever
// the aspect ratio (a 1×16M image would otherwise need a 256×16M buffer).
func resizeImage(img image.Image, outW, outH int) []uint8 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	row := make([]uint8, w*3)
	hrow := row
	var hb []int
	var hk [][]int32
	if outW != w {
		hb, hk = coeffs(w, outW)
		hrow = make([]uint8, outW*3)
	}
	out := make([]uint8, outW*outH*3)
	if outH == h {
		for y := 0; y < h; y++ {
			dst := out[y*outW*3 : (y+1)*outW*3]
			if outW == w {
				rgbRow(dst, img, b.Min.Y+y)
				continue
			}
			rgbRow(row, img, b.Min.Y+y)
			resampleRow(dst, row, hb, hk)
		}
		return out
	}
	vb, vk := coeffs(h, outH)
	acc := make([]int32, outW*outH*3)
	for i := range acc {
		acc[i] = 1 << (precisionBits - 1)
	}
	first := 0 // first output row whose window has not ended
	for y := 0; y < h; y++ {
		rgbRow(row, img, b.Min.Y+y)
		if outW != w {
			resampleRow(hrow, row, hb, hk)
		}
		for first < outH && vb[first]+len(vk[first]) <= y {
			first++
		}
		for yy := first; yy < outH && vb[yy] <= y; yy++ {
			d := y - vb[yy]
			if d >= len(vk[yy]) {
				continue
			}
			kv := vk[yy][d]
			a := acc[yy*outW*3 : (yy+1)*outW*3]
			for i, v := range hrow {
				a[i] += int32(v) * kv
			}
		}
	}
	for i, v := range acc {
		out[i] = clip8(v)
	}
	return out
}

// resampleRow resamples one row of RGB bytes horizontally.
func resampleRow(dst, row []uint8, bounds []int, k [][]int32) {
	for xx := range bounds {
		for c := 0; c < 3; c++ {
			ss := int32(1 << (precisionBits - 1))
			for x, kv := range k[xx] {
				ss += int32(row[(bounds[xx]+x)*3+c]) * kv
			}
			dst[xx*3+c] = clip8(ss)
		}
	}
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
			resampleRow(dst[y*out*3:(y+1)*out*3], src[y*w*3:(y+1)*w*3], bounds, k)
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

// MaxImageSide bounds each side of an image to preprocess. PIL's bilinear
// filter widens with the scale: each output row of a 1×16M image would
// read 131,000 source rows, 25 billion operations. At 16,384 pixels a side,
// the worst case stays near 100 million.
const MaxImageSide = 1 << 14

// Preprocess turns an image into the vision tower's input: RGB, resized to
// ImageSize × ImageSize with PIL's bilinear filter, normalized.
func (c Config) Preprocess(img image.Image) ([]float32, error) {
	return c.PreprocessInto(nil, img)
}

// PreprocessInto is Preprocess writing the pixels into dst when it is
// large enough.
func (c Config) PreprocessInto(dst []float32, img image.Image) ([]float32, error) {
	if b := img.Bounds(); b.Dx() < 1 || b.Dy() < 1 || b.Dx() > MaxImageSide || b.Dy() > MaxImageSide {
		return nil, fmt.Errorf("siglip: image of %d×%d pixels, sides from 1 to %d", b.Dx(), b.Dy(), MaxImageSide)
	}
	return PixelsInto(dst, resizeImage(img, c.ImageSize, c.ImageSize), c.ImageSize)
}
