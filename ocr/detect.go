package ocr

import (
	"image"
	"math"
	"sort"
	"sync"

	"github.com/bornholm/indecis/internal/linedet"
)

// Detector finds the text lines of a page. It is safe for concurrent use.
type Detector struct {
	m *linedet.Model

	// LongSide is the length the page's longer side is resized to (1024,
	// as Indra OCR).
	LongSide int
	// Threshold on the text probability (0.5).
	Threshold float32

	tapes sync.Pool // *linedet.Tape
}

// LoadDetector reads a detector from dir: config.json and
// model.safetensors.
func LoadDetector(dir string) (*Detector, error) {
	m, err := linedet.Load(dir)
	if err != nil {
		return nil, err
	}
	return NewDetector(m), nil
}

// NewDetector wraps a line detection model.
func NewDetector(m *linedet.Model) *Detector {
	return &Detector{m: m, LongSide: 1024, Threshold: 0.5}
}

// Model returns the detection model.
func (d *Detector) Model() *linedet.Model { return d.m }

// detInput is a page resized for the detector: ink [H, W], sides multiples
// of 32, and the scale from page pixels to input pixels.
type detInput struct {
	ink   []float32
	H, W  int
	scale float64
}

// prepare resizes page so that its longer side is longSide, as ink, padded
// with paper to multiples of 32.
func prepare(page *image.Gray, longSide int) detInput {
	b := page.Bounds()
	sw, sh := b.Dx(), b.Dy()
	scale := float64(longSide) / float64(max(sw, sh))
	dw, dh := max(1, int(math.Round(float64(sw)*scale))), max(1, int(math.Round(float64(sh)*scale)))
	src := make([]uint8, sw*sh)
	for y := range sh {
		copy(src[y*sw:(y+1)*sw], page.Pix[page.PixOffset(b.Min.X, b.Min.Y+y):])
	}
	var r []float32
	if scale < 1 {
		r = resizeArea(src, sw, sh, dw, dh)
	} else {
		r = resizeLinear(src, sw, sh, dw, dh)
	}
	H := (dh + linedet.Multiple - 1) / linedet.Multiple * linedet.Multiple
	W := (dw + linedet.Multiple - 1) / linedet.Multiple * linedet.Multiple
	ink := make([]float32, H*W)
	for y := range dh {
		for x := range dw {
			ink[y*W+x] = 1 - r[y*dw+x]/255
		}
	}
	return detInput{ink: ink, H: H, W: W, scale: scale}
}

// Detect returns the boxes of the page's lines, top to bottom. When the
// lines come out small (a dense page), the page is read again larger.
func (d *Detector) Detect(page *image.Gray) []Box {
	in := prepare(page, d.LongSide)
	boxes := d.detect(in)
	if len(boxes) >= 8 {
		hs := make([]float64, len(boxes))
		for i, b := range boxes {
			hs[i] = (b.Y1 - b.Y0) * 2 // input pixels
		}
		sort.Float64s(hs)
		if med := hs[len(hs)/2]; med < adaptiveMinH {
			long := int(min(adaptiveMaxSide, float64(d.LongSide)*adaptiveTargetH/max(med, 1))) / linedet.Multiple * linedet.Multiple
			if long > d.LongSide {
				in = prepare(page, long)
				boxes = d.detect(in)
			}
		}
	}
	// Output pixels are 2 input pixels; input pixels are 1/scale page
	// pixels.
	f := 2 / in.scale
	b := page.Bounds()
	for i := range boxes {
		bx := &boxes[i]
		bx.X0 = max(float64(b.Min.X), float64(b.Min.X)+bx.X0*f)
		bx.Y0 = max(float64(b.Min.Y), float64(b.Min.Y)+bx.Y0*f)
		bx.X1 = min(float64(b.Max.X), float64(b.Min.X)+bx.X1*f)
		bx.Y1 = min(float64(b.Max.Y), float64(b.Min.Y)+bx.Y1*f)
	}
	sortLines(boxes)
	return boxes
}

// Small lines are read again at a larger scale (Indra OCR's rule): when
// the median line is below adaptiveMinH input pixels, the page is resized
// for adaptiveTargetH, up to adaptiveMaxSide.
const (
	adaptiveMinH    = 9
	adaptiveTargetH = 12
	adaptiveMaxSide = 2048
)

func (d *Detector) detect(in detInput) []Box {
	t, _ := d.tapes.Get().(*linedet.Tape)
	if t == nil {
		t = new(linedet.Tape)
	}
	defer d.tapes.Put(t)
	out := d.m.Forward(t, in.ink, in.H, in.W, false)
	return mergeFragments(d.decode(out.V, out.S.H, out.S.W))
}

// decode turns the output maps [H, W, 3] into line boxes in output
// pixels: each connected core above the threshold, its top and bottom the
// mean of its pixels' predictions, its ends grown back.
func (d *Detector) decode(maps []float32, H, W int) []Box {
	thr := float32(math.Log(float64(d.Threshold) / (1 - float64(d.Threshold)))) // on the logit
	label := make([]int32, H*W)
	var boxes []Box
	var stack []int
	next := int32(0)
	for start := range H * W {
		if label[start] != 0 || maps[start*3] <= thr {
			continue
		}
		next++
		label[start] = next
		stack = append(stack[:0], start)
		x0, x1 := W, 0
		var top, bot float64
		n := 0
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			y, x := p/W, p%W
			x0, x1 = min(x0, x), max(x1, x+1)
			yc := float64(y) + 0.5
			top += yc - float64(d.m.Dist(maps[p*3+1]))
			bot += yc + float64(d.m.Dist(maps[p*3+2]))
			n++
			for _, q := range [4]int{p - 1, p + 1, p - W, p + W} {
				if q < 0 || q >= H*W || (q == p-1 && x == 0) || (q == p+1 && x == W-1) {
					continue
				}
				if label[q] == 0 && maps[q*3] > thr {
					label[q] = next
					stack = append(stack, q)
				}
			}
		}
		if n < 2 {
			continue
		}
		y0, y1 := top/float64(n), bot/float64(n)
		lh := max(0.5, y1-y0)
		ex := min(coreShrinkX*lh, 0.5*lh)
		boxes = append(boxes, Box{X0: float64(x0) - ex, Y0: y0, X1: float64(x1) + ex, Y1: y1})
	}
	return boxes
}

// A line's core, the detector's target, is coreHeight of its height around
// its middle, without coreShrinkX of its height at each end (Indra OCR's
// geometry): neighbouring lines' cores never touch.
const (
	coreHeight  = 0.4
	coreShrinkX = 0.15
)

// sortLines orders boxes top to bottom, then left to right within a band:
// boxes sorted by their centers, a band grows while the next center lies
// within half the band's tallest box of the band's first center. Two
// passes rather than one comparator, which a pair-dependent band would make
// intransitive.
func sortLines(b []Box) {
	center := func(x Box) float64 { return (x.Y0 + x.Y1) / 2 }
	sort.SliceStable(b, func(i, j int) bool { return center(b[i]) < center(b[j]) })
	for start := 0; start < len(b); {
		end, tall := start+1, b[start].Y1-b[start].Y0
		for end < len(b) && center(b[end])-center(b[start]) < tall/2 {
			tall = max(tall, b[end].Y1-b[end].Y0)
			end++
		}
		band := b[start:end]
		sort.SliceStable(band, func(i, j int) bool { return band[i].X0 < band[j].X0 })
		start = end
	}
}

// ReadPage finds the lines of a page and reads them.
func ReadPage(d *Detector, r *Recognizer, page *image.Gray) ([]Box, []Line) {
	boxes := d.Detect(page)
	return boxes, r.ReadLines(page, boxes)
}

// IoU is the intersection over union of two boxes.
func IoU(a, b Box) float64 {
	iw := min(a.X1, b.X1) - max(a.X0, b.X0)
	ih := min(a.Y1, b.Y1) - max(a.Y0, b.Y0)
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	return inter / ((a.X1-a.X0)*(a.Y1-a.Y0) + (b.X1-b.X0)*(b.Y1-b.Y0) - inter)
}

// Match pairs predicted boxes with reference boxes, best IoU first, one to
// one, above minIoU: match[i] is the prediction of reference i, or -1.
func Match(ref, pred []Box, minIoU float64) []int {
	type pair struct {
		i, j int
		iou  float64
	}
	var ps []pair
	for i, r := range ref {
		for j, p := range pred {
			if v := IoU(r, p); v >= minIoU {
				ps = append(ps, pair{i, j, v})
			}
		}
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].iou > ps[b].iou })
	match := make([]int, len(ref))
	for i := range match {
		match[i] = -1
	}
	used := make([]bool, len(pred))
	for _, p := range ps {
		if match[p.i] < 0 && !used[p.j] {
			match[p.i] = p.j
			used[p.j] = true
		}
	}
	return match
}

// DetectionScore counts matched lines.
type DetectionScore struct {
	Ref, Pred, Matched int
}

// Precision is the share of predicted lines matched.
func (s DetectionScore) Precision() float64 { return float64(s.Matched) / float64(max(1, s.Pred)) }

// Recall is the share of reference lines matched.
func (s DetectionScore) Recall() float64 { return float64(s.Matched) / float64(max(1, s.Ref)) }

// F1 is the harmonic mean of precision and recall.
func (s DetectionScore) F1() float64 {
	p, r := s.Precision(), s.Recall()
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// mergeFragments joins boxes of the same line that the core map left
// apart (at word gaps): boxes that overlap vertically over most of the
// smaller height and are less than half a height apart (table cells are
// further apart).
func mergeFragments(b []Box) []Box {
	for merged := true; merged; {
		merged = false
		for i := 0; i < len(b) && !merged; i++ {
			for j := i + 1; j < len(b); j++ {
				hi, hj := b[i].Y1-b[i].Y0, b[j].Y1-b[j].Y0
				h := min(hi, hj)
				over := min(b[i].Y1, b[j].Y1) - max(b[i].Y0, b[j].Y0)
				gap := max(b[i].X0, b[j].X0) - min(b[i].X1, b[j].X1)
				if over > 0.6*h && gap < 0.5*h && max(hi, hj) < 1.6*h {
					b[i] = Box{min(b[i].X0, b[j].X0), min(b[i].Y0, b[j].Y0), max(b[i].X1, b[j].X1), max(b[i].Y1, b[j].Y1)}
					b = append(b[:j], b[j+1:]...)
					merged = true
					break
				}
			}
		}
	}
	return b
}
