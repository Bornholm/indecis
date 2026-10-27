package main

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"

	"github.com/bornholm/indecis/ocr"
)

// pageSample is a rendered page and the ink box and text of each line.
type pageSample struct {
	page  *image.Gray
	boxes []ocr.Box
	texts []string
}

// page composes a page for detection training: one or two columns of
// headings, paragraphs, lists and tables, then the scan damage of lines.
func (g *generator) page(r *rand.Rand) pageSample {
	W := 900 + r.Intn(900)
	H := int(float64(W) * (1.3 + 0.15*r.Float64()))
	paper, ink := uint8(205+r.Intn(51)), uint8(r.Intn(80))
	img := image.NewGray(image.Rect(0, 0, W, H))
	fill(img, paper)
	src := image.NewUniform(color.Gray{ink})
	ps := pageSample{page: img}

	body := g.fonts.pick(r)
	base := max(10, int(float64(W)*(0.010+0.012*r.Float64()))) // body text size, pixels
	margin := int(float64(W) * (0.05 + 0.07*r.Float64()))
	cols := 1
	if r.Float64() < 0.3 {
		cols = 2
	}
	gutter := base * 2
	colW := (W - 2*margin - (cols-1)*gutter) / cols
	y := margin + r.Intn(margin+1)
	col := 0
	x0 := margin

	// line draws one line of text at (x, y baseline), records its box.
	line := func(f *fontFile, size int, s string, x, y int) bool {
		if s == "" || !f.draws(s) {
			return false
		}
		face, err := g.faces.get(f, size)
		if err != nil {
			return false
		}
		if _, ok := metrics(face); !ok {
			return false
		}
		d := font.Drawer{Dst: img, Src: src, Face: face, Dot: fixed.P(x, y)}
		b, _ := d.BoundString(s)
		d.DrawString(s)
		box := ocr.Box{X0: float64(b.Min.X.Floor()), Y0: float64(b.Min.Y.Floor()), X1: float64(b.Max.X.Ceil()), Y1: float64(b.Max.Y.Ceil())}
		if box.X1-box.X0 < 2 || box.Y1-box.Y0 < 2 {
			return false
		}
		ps.boxes = append(ps.boxes, box)
		ps.texts = append(ps.texts, s)
		return true
	}
	width := func(f *fontFile, size int, s string) int {
		face, err := g.faces.get(f, size)
		if err != nil {
			return 1 << 30
		}
		return font.MeasureString(face, s).Ceil()
	}
	// wrap breaks running text into lines at most w pixels wide.
	wrap := func(f *fontFile, size int, text string, w int) []string {
		var out []string
		cur := ""
		for _, word := range strings.Fields(text) {
			try := word
			if cur != "" {
				try = cur + " " + word
			}
			if width(f, size, try) > w && cur != "" {
				out = append(out, cur)
				cur = word
			} else {
				cur = try
			}
		}
		if cur != "" {
			out = append(out, cur)
		}
		return out
	}
	nextCol := func() bool {
		col++
		if col >= cols {
			return false
		}
		x0 = margin + col*(colW+gutter)
		y = margin
		return true
	}
	lead := func(size int) int { return int(float64(size) * (1.15 + 0.35*r.Float64())) }

	for blocks := 0; blocks < 40; blocks++ {
		if y > H-margin-3*base && !nextCol() {
			break
		}
		switch k := r.Float64(); {
		case k < 0.12: // heading
			size := int(float64(base) * (1.2 + 0.8*r.Float64()))
			f := g.fonts.pick(r)
			ls := wrap(f, size, g.text.line(r), colW)
			if len(ls) == 0 {
				continue
			}
			y += lead(size)
			line(f, size, ls[0], x0, y)
			y += size / 2
		case k < 0.6: // paragraph
			var text strings.Builder
			for range 2 + r.Intn(5) {
				text.WriteString(g.text.line(r) + " ")
			}
			indent := 0
			if r.Float64() < 0.5 {
				indent = base * 2
			}
			ld := lead(base)
			for i, l := range wrap(body, base, text.String(), colW-indent) {
				y += ld
				if y > H-margin {
					if !nextCol() {
						break
					}
					y += ld
				}
				x := x0
				if i == 0 {
					x += indent
				}
				line(body, base, l, x, y)
			}
			y += ld / 2
		case k < 0.75: // list
			ld := lead(base)
			for range 2 + r.Intn(5) {
				y += ld
				if y > H-margin {
					break
				}
				bullet := pick(r, []string{"•", "-", "–", "*", "1.", "a)"})
				s := bullet + " " + g.text.line(r)
				ls := wrap(body, base, s, colW-base)
				if len(ls) > 0 {
					line(body, base, ls[0], x0+base, y)
				}
			}
			y += ld / 2
		default: // table: a grid of short cells
			rows, ncols := 2+r.Intn(6), 2+r.Intn(4)
			cellW := colW / ncols
			cellH := int(float64(base) * (1.8 + 0.8*r.Float64()))
			if y+rows*cellH > H-margin {
				continue
			}
			ruled := r.Float64() < 0.7
			for i := 0; i <= rows && ruled; i++ {
				hline(img, x0, x0+ncols*cellW, y+i*cellH, 1, ink)
			}
			for j := 0; j <= ncols && ruled; j++ {
				vline(img, x0+j*cellW, y, y+rows*cellH, 1, ink)
			}
			for i := range rows {
				for j := range ncols {
					if r.Float64() < 0.2 {
						continue // empty cell
					}
					s := generated(r, g.text.charset)
					if r.Float64() < 0.5 {
						s = strings.Fields(s + " x")[0]
					}
					for len(s) > 1 && width(body, base, s) > cellW-base/2 {
						s = s[:len(s)-1]
					}
					s = strings.TrimSpace(s)
					line(body, base, s, x0+j*cellW+base/3, y+i*cellH+cellH/2+base/3)
				}
			}
			y += rows*cellH + base
		}
	}
	// Page number.
	if r.Float64() < 0.6 {
		line(body, base, digitsN(r), W/2, H-margin/2)
	}

	if r.Float64() < 0.3 {
		angle := (2*r.Float64() - 1) * 1.0 * math.Pi / 180
		ps.page = rotate(img, angle, paper)
		cx, cy := float64(W)/2, float64(H)/2
		cos, sin := math.Cos(angle), math.Sin(angle)
		for i, b := range ps.boxes {
			// rotate maps a destination point (x, y) to the source point
			// R(-angle); a source point moves by R(angle) the other way.
			x0, y0, x1, y1 := math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
			for _, p := range [][2]float64{{b.X0, b.Y0}, {b.X1, b.Y0}, {b.X0, b.Y1}, {b.X1, b.Y1}} {
				dx, dy := p[0]-cx, p[1]-cy
				x := cos*dx - sin*dy + cx
				yy := sin*dx + cos*dy + cy
				x0, x1 = min(x0, x), max(x1, x)
				y0, y1 = min(y0, yy), max(y1, yy)
			}
			ps.boxes[i] = ocr.Box{X0: x0, Y0: y0, X1: x1, Y1: y1}
		}
	}
	degrade(ps.page, r, paper, ink, map[string]string{})
	return ps
}

func digitsN(r *rand.Rand) string {
	return strings.TrimLeft(digits(r, 1+r.Intn(3)), "0") + "1"
}
