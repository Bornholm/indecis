package main

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"math"
	"math/rand"
	"strings"
	"unicode"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"

	"github.com/bornholm/indecis/ocr"
)

// generator draws synthetic lines. One per worker: the face cache is not
// safe for concurrent use.
type generator struct {
	fonts *fontSet
	text  *texter
	faces faces
	crop  ocr.CropOptions
}

// sample is a rendered line: the patch it sits in (with its neighbours),
// its ink box on the patch, its text.
type sample struct {
	page *image.Gray
	box  ocr.Box
	text string
	meta map[string]string
}

func (g *generator) sample(r *rand.Rand) sample {
	for {
		if s, ok := g.try(r); ok {
			return s
		}
	}
}

func (g *generator) try(r *rand.Rand) (sample, bool) {
	text := g.text.line(r)
	var f *fontFile
	for range 20 {
		if c := g.fonts.pick(r); c.draws(text) {
			f = c
			break
		}
	}
	if f == nil {
		return sample{}, false
	}
	var above, below string
	if r.Float64() < 0.7 {
		if above = g.text.line(r); !f.draws(above) {
			above = ""
		}
	}
	if r.Float64() < 0.7 {
		if below = g.text.line(r); !f.draws(below) {
			below = ""
		}
	}
	size := 12 + r.Intn(45)
	face, err := g.faces.get(f, size)
	if err != nil {
		return sample{}, false
	}
	m, ok := metrics(face)
	if !ok {
		if !f.broken.Swap(true) {
			log.Printf("%s: x/image cannot read its metrics, dropped", f.name)
		}
		return sample{}, false
	}
	lineH := float64((m.Ascent + m.Descent).Ceil())
	spacing := lineH * (0.95 + 0.7*r.Float64())
	tracking := 0.0
	if r.Float64() < 0.15 {
		tracking = float64(size) * (0.25*r.Float64() - 0.05)
	}

	// Columns: some spaces drawn wider than one, the text keeps one.
	drawn := text
	if r.Float64() < 0.12 {
		var b strings.Builder
		for _, c := range text {
			b.WriteRune(c)
			if c == ' ' && r.Float64() < 0.5 {
				b.WriteString(strings.Repeat(" ", 1+r.Intn(5)))
			}
		}
		drawn = b.String()
	}
	// Old print: the long s inside words, read as s (transcriptions
	// modernize it).
	if r.Float64() < 0.04 && f.covers['ſ'] {
		drawn = longS(drawn)
	}
	width := func(s string) float64 {
		return float64(font.MeasureString(face, s).Ceil()) + tracking*float64(len([]rune(s)))
	}
	margin := float64(size) * (0.5 + 1.5*r.Float64())
	textW := max(width(drawn), width(above), width(below))
	W := int(textW + 2*margin + 2)
	H := int(3*spacing + 2*margin)
	if W > 6000 || W <= 0 || H <= 0 {
		return sample{}, false
	}

	paper, ink := uint8(200+r.Intn(56)), uint8(r.Intn(90))
	dark := r.Float64() < 0.05
	if dark {
		paper, ink = uint8(r.Intn(70)), uint8(180+r.Intn(76))
	}
	page := image.NewGray(image.Rect(0, 0, W, H))
	fill(page, paper)
	mask := image.NewAlpha(page.Rect)
	src := image.NewUniform(color.Gray{ink})

	x0 := margin
	draw := func(dst draw.Image, src image.Image, s string, x, y float64) {
		d := font.Drawer{Dst: dst, Src: src, Face: face, Dot: fixed.Point26_6{X: fixed.I(int(x)), Y: fixed.I(int(y))}}
		if tracking == 0 {
			d.DrawString(s)
			return
		}
		for _, c := range s {
			d.DrawString(string(c))
			d.Dot.X += fixed.Int26_6(tracking * 64)
		}
	}
	base := margin + spacing + float64(m.Ascent.Ceil())
	jitter := func() float64 { return x0 + float64(size)*(r.Float64()-0.3)*0.5 }
	if above != "" {
		draw(page, src, above, jitter(), base-spacing)
	}
	draw(page, src, drawn, x0, base)
	draw(mask, image.Opaque, drawn, x0, base)
	if below != "" {
		draw(page, src, below, jitter(), base+spacing)
	}
	box, ok := inkBox(mask)
	if !ok {
		return sample{}, false
	}
	meta := map[string]string{"font": f.name}

	// Rules near the line: underline, table borders.
	if r.Float64() < 0.08 {
		y := int(box.Y1) + 1 + r.Intn(max(1, size/6))
		hline(page, int(box.X0)-r.Intn(size), int(box.X1)+r.Intn(size), y, 1+r.Intn(2), ink)
	}
	if r.Float64() < 0.05 {
		gap := float64(size) * (0.3 + r.Float64())
		for _, x := range []float64{box.X0 - gap, box.X1 + gap} {
			vline(page, int(x), int(box.Y0-spacing), int(box.Y1+spacing), 1+r.Intn(2), ink)
		}
	}

	if r.Float64() < 0.25 {
		angle := (2*r.Float64() - 1) * 1.5 * math.Pi / 180
		page = rotate(page, angle, paper)
		m2 := image.NewGray(mask.Rect)
		for i, a := range mask.Pix {
			m2.Pix[i] = 255 - a
		}
		m2 = rotate(m2, angle, 255)
		for i, v := range m2.Pix {
			mask.Pix[i] = 255 - v
		}
		if box, ok = inkBox(mask); !ok {
			return sample{}, false
		}
	}
	if r.Float64() < 0.05 {
		dotMatrix(page, max(3, size/7), paper)
		meta["dots"] = "1"
	}
	degrade(page, r, paper, ink, meta)

	// The box as a detector would give it: a little off on each side.
	h := box.Y1 - box.Y0
	j := func() float64 { return (2*r.Float64() - 1) * 0.08 * h }
	box = ocr.Box{X0: box.X0 + j(), Y0: box.Y0 + j(), X1: box.X1 + j(), Y1: box.Y1 + j()}
	return sample{page: page, box: box, text: text, meta: meta}, true
}

// inkBox is the bounding box of the mask's covered pixels.
func inkBox(m *image.Alpha) (ocr.Box, bool) {
	b := m.Rect
	x0, y0, x1, y1 := b.Max.X, b.Max.Y, b.Min.X, b.Min.Y
	for y := b.Min.Y; y < b.Max.Y; y++ {
		row := m.Pix[(y-b.Min.Y)*m.Stride : (y-b.Min.Y)*m.Stride+b.Dx()]
		for x, a := range row {
			if a > 64 {
				x0, x1 = min(x0, x), max(x1, x+1)
				y0, y1 = min(y0, y), max(y1, y+1)
			}
		}
	}
	if x1 <= x0 {
		return ocr.Box{}, false
	}
	return ocr.Box{X0: float64(x0), Y0: float64(y0), X1: float64(x1), Y1: float64(y1)}, true
}

func fill(g *image.Gray, v uint8) {
	for i := range g.Pix {
		g.Pix[i] = v
	}
}

func hline(g *image.Gray, x0, x1, y, t int, v uint8) {
	for yy := y; yy < y+t; yy++ {
		for x := x0; x < x1; x++ {
			if (image.Point{x, yy}).In(g.Rect) {
				g.Pix[g.PixOffset(x, yy)] = v
			}
		}
	}
}

func vline(g *image.Gray, x, y0, y1, t int, v uint8) {
	for xx := x; xx < x+t; xx++ {
		for y := y0; y < y1; y++ {
			if (image.Point{xx, y}).In(g.Rect) {
				g.Pix[g.PixOffset(xx, y)] = v
			}
		}
	}
}

// rotate turns g by angle (radians) about its center, bilinearly, bg
// outside.
func rotate(g *image.Gray, angle float64, bg uint8) *image.Gray {
	W, H := g.Rect.Dx(), g.Rect.Dy()
	out := image.NewGray(g.Rect)
	cx, cy := float64(W)/2, float64(H)/2
	cos, sin := math.Cos(angle), math.Sin(angle)
	at := func(x, y int) float64 {
		if x < 0 || y < 0 || x >= W || y >= H {
			return float64(bg)
		}
		return float64(g.Pix[y*g.Stride+x])
	}
	for y := range H {
		for x := range W {
			dx, dy := float64(x)+0.5-cx, float64(y)+0.5-cy
			sx := cos*dx + sin*dy + cx - 0.5
			sy := -sin*dx + cos*dy + cy - 0.5
			ix, iy := int(math.Floor(sx)), int(math.Floor(sy))
			fx, fy := sx-float64(ix), sy-float64(iy)
			v := (at(ix, iy)*(1-fx)+at(ix+1, iy)*fx)*(1-fy) + (at(ix, iy+1)*(1-fx)+at(ix+1, iy+1)*fx)*fy
			out.Pix[y*out.Stride+x] = uint8(v + 0.5)
		}
	}
	return out
}

// degrade applies the photometric damage of scans, faxes and photos, each
// with its own probability.
func degrade(g *image.Gray, r *rand.Rand, paper, ink uint8, meta map[string]string) {
	W, H := g.Rect.Dx(), g.Rect.Dy()
	if r.Float64() < 0.2 { // uneven lighting
		a, b := (2*r.Float64()-1)*40, (2*r.Float64()-1)*40
		for y := range H {
			for x := range W {
				d := a*float64(x)/float64(W) + b*float64(y)/float64(H)
				g.Pix[y*g.Stride+x] = clamp8(float64(g.Pix[y*g.Stride+x]) + d)
			}
		}
	}
	if r.Float64() < 0.1 { // strokes thickened or thinned
		morph(g, r.Intn(2) == 0)
	}
	if r.Float64() < photocopy {
		copyDamage(g, r, paper)
		meta["copy"] = "1"
	}
	if r.Float64() < 0.3 { // low resolution
		s := 0.35 + 0.5*r.Float64()
		w, h := max(1, int(float64(W)*s)), max(1, int(float64(H)*s))
		small := image.NewGray(image.Rect(0, 0, w, h))
		xdraw.ApproxBiLinear.Scale(small, small.Rect, g, g.Rect, xdraw.Src, nil)
		xdraw.BiLinear.Scale(g, g.Rect, small, small.Rect, xdraw.Src, nil)
	}
	if r.Float64() < 0.4 {
		blur(g, 0.3+0.9*r.Float64())
	}
	if r.Float64() < 0.5 {
		sigma := 2 + 14*r.Float64()
		for i, v := range g.Pix {
			g.Pix[i] = clamp8(float64(v) + r.NormFloat64()*sigma)
		}
	}
	if r.Float64() < 0.08 { // fax: binarized
		t := (float64(paper) + float64(ink)) / 2
		for i, v := range g.Pix {
			if float64(v) < t {
				g.Pix[i] = 0
			} else {
				g.Pix[i] = 255
			}
		}
		meta["binary"] = "1"
	}
	if r.Float64() < 0.3 {
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, g, &jpeg.Options{Quality: 15 + r.Intn(70)}); err == nil {
			if img, err := jpeg.Decode(&buf); err == nil {
				// The decoder rounds its rows up to whole blocks: copy
				// row by row.
				if gg, ok := img.(*image.Gray); ok {
					for y := range H {
						copy(g.Pix[y*g.Stride:y*g.Stride+W], gg.Pix[y*gg.Stride:])
					}
				}
			}
		}
	}
}

func clamp8(v float64) uint8 { return uint8(min(max(v+0.5, 0), 255)) }

// photocopy is the share of lines damaged as photocopies and old
// typewritten forms are (-copy).
var photocopy = 0.0

// copyDamage spoils a line as copies of copies do: the ink faded and
// broken by blotches, stains of the paper, specks.
func copyDamage(g *image.Gray, r *rand.Rand, paper uint8) {
	W, H := g.Rect.Dx(), g.Rect.Dy()
	p := float64(paper)
	fade := 0.3 + 0.5*r.Float64() // share of the ink kept at the most faded
	field := smoothField(r, W, H, 3+r.Intn(5))
	stains := smoothField(r, W, H, 20+r.Intn(40))
	stainAmp := 50 * r.Float64()
	for y := range H {
		for x := range W {
			i := y*g.Stride + x
			v := float64(g.Pix[i])
			k := 1 - (1-fade)*field[y*W+x] // ink kept here
			v = p - (p-v)*k - stainAmp*stains[y*W+x]
			g.Pix[i] = clamp8(v)
		}
	}
	for range int(float64(W*H) * 0.002 * r.Float64()) {
		x, y, sz := r.Intn(W), r.Intn(H), 1+r.Intn(2)
		v := uint8(r.Intn(120))
		for dy := range sz {
			for dx := range sz {
				if x+dx < W && y+dy < H {
					g.Pix[(y+dy)*g.Stride+x+dx] = min(g.Pix[(y+dy)*g.Stride+x+dx], v)
				}
			}
		}
	}
}

// smoothField is a random field [h, w] in [0, 1], smooth over cell pixels.
func smoothField(r *rand.Rand, w, h, cell int) []float64 {
	gw, gh := w/cell+2, h/cell+2
	grid := make([]float64, gw*gh)
	for i := range grid {
		grid[i] = r.Float64()
	}
	f := make([]float64, w*h)
	for y := range h {
		gy := float64(y) / float64(cell)
		y0 := int(gy)
		fy := gy - float64(y0)
		for x := range w {
			gx := float64(x) / float64(cell)
			x0 := int(gx)
			fx := gx - float64(x0)
			a := grid[y0*gw+x0]*(1-fx) + grid[y0*gw+x0+1]*fx
			b := grid[(y0+1)*gw+x0]*(1-fx) + grid[(y0+1)*gw+x0+1]*fx
			f[y*w+x] = a*(1-fy) + b*fy
		}
	}
	return f
}

// morph applies a 3×3 minimum (darkest) or maximum filter.
func morph(g *image.Gray, darkest bool) {
	W, H := g.Rect.Dx(), g.Rect.Dy()
	src := append([]uint8(nil), g.Pix...)
	for y := range H {
		for x := range W {
			v := src[y*g.Stride+x]
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= W || yy >= H {
						continue
					}
					u := src[yy*g.Stride+xx]
					if darkest {
						v = min(v, u)
					} else {
						v = max(v, u)
					}
				}
			}
			g.Pix[y*g.Stride+x] = v
		}
	}
}

// blur is a separable Gaussian blur.
func blur(g *image.Gray, sigma float64) {
	rad := int(math.Ceil(2.5 * sigma))
	k := make([]float64, 2*rad+1)
	var sum float64
	for i := range k {
		d := float64(i - rad)
		k[i] = math.Exp(-d * d / (2 * sigma * sigma))
		sum += k[i]
	}
	for i := range k {
		k[i] /= sum
	}
	W, H := g.Rect.Dx(), g.Rect.Dy()
	tmp := make([]float64, W*H)
	for y := range H {
		for x := range W {
			var v float64
			for i, w := range k {
				xx := min(max(x+i-rad, 0), W-1)
				v += w * float64(g.Pix[y*g.Stride+xx])
			}
			tmp[y*W+x] = v
		}
	}
	for y := range H {
		for x := range W {
			var v float64
			for i, w := range k {
				yy := min(max(y+i-rad, 0), H-1)
				v += w * tmp[yy*W+x]
			}
			g.Pix[y*g.Stride+x] = clamp8(v)
		}
	}
}

func metrics(face font.Face) (m font.Metrics, ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	return face.Metrics(), true
}

// dotMatrix keeps the ink only inside round dots on a k-pixel grid, as a
// dot-matrix or worn thermal printer prints.
func dotMatrix(g *image.Gray, k int, paper uint8) {
	c := float64(k-1) / 2
	rad2 := (0.42 * float64(k)) * (0.42 * float64(k))
	for y := range g.Rect.Dy() {
		for x := range g.Rect.Dx() {
			dx, dy := float64(x%k)-c, float64(y%k)-c
			if dx*dx+dy*dy > rad2 {
				g.Pix[y*g.Stride+x] = paper
			}
		}
	}
	blur(g, 0.6)
}

// longS writes a lowercase s that does not end a word as ſ, as printers did
// until the 19th century.
func longS(s string) string {
	rs := []rune(s)
	for i, c := range rs {
		if c == 's' && i+1 < len(rs) && unicode.IsLetter(rs[i+1]) {
			rs[i] = 'ſ'
		}
	}
	return string(rs)
}
