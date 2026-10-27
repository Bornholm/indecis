package ocr

import (
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linedet"
	"github.com/bornholm/indecis/internal/optim"
)

// DetTrainOptions sets a detector training run.
type DetTrainOptions struct {
	Steps       int
	Batch       int // crops per step
	LR          float64
	Warmup      int
	WeightDecay float64
	Clip        float64
	Slots       int // see TrainOptions.Slots; default 8

	Crop             int // side of the training crops (512)
	MinSide, MaxSide int // the page's longer side is drawn in [MinSide, MaxSide] (640, 1280)
	Seed             int64

	Eval      []PageSample // pages measured every EvalEvery steps
	EvalEvery int
	Out       string // the best model on Eval (F1), or the last one
	LogEvery  int
	Log       io.Writer
}

// cachedPage is a training page decoded once, longer side MaxSide.
type cachedPage struct {
	pix   []uint8
	W, H  int
	boxes []Box // in cached pixels
}

// TrainDetector trains m on labeled pages.
func TrainDetector(ctx context.Context, m *linedet.Model, pages []PageSample, o DetTrainOptions) (DetectionScore, error) {
	if o.Slots <= 0 {
		o.Slots = 8
	}
	if o.Crop == 0 {
		o.Crop = 512
	}
	if o.MinSide == 0 {
		o.MinSide = 640
	}
	if o.MaxSide == 0 {
		o.MaxSide = max(1280, o.MinSide)
	}
	if o.Log == nil {
		o.Log = io.Discard
	}
	o.Crop = o.Crop / linedet.Multiple * linedet.Multiple
	switch {
	case o.Steps <= 0 || o.Batch <= 0:
		return DetectionScore{}, fmt.Errorf("ocr: Steps and Batch must be positive")
	case len(pages) == 0:
		return DetectionScore{}, fmt.Errorf("ocr: no training page")
	case o.Crop < linedet.Multiple:
		return DetectionScore{}, fmt.Errorf("ocr: Crop must be at least %d pixels", linedet.Multiple)
	case o.MinSide <= 0 || o.MaxSide < o.MinSide:
		return DetectionScore{}, fmt.Errorf("ocr: invalid page sides [%d, %d]", o.MinSide, o.MaxSide)
	}

	cache := make([]cachedPage, len(pages))
	errs := make([]error, len(pages))
	linalg.Parallel(len(pages), 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			g, err := ReadImage(pages[i].Image)
			if err != nil {
				errs[i] = err
				continue
			}
			cache[i] = cachePage(g, pages[i], o.MaxSide)
		}
	})
	for _, err := range errs {
		if err != nil {
			return DetectionScore{}, err
		}
	}
	fmt.Fprintf(o.Log, "%d training pages decoded\n", len(pages))

	type slot struct {
		g    *linedet.Grads
		t    linedet.Tape
		d    []float32
		loss float64
	}
	slots := make([]*slot, o.Slots)
	for i := range slots {
		slots[i] = &slot{g: m.NewGrads()}
	}
	total := m.NewGrads()
	dense := m.Dense(total)
	cfg := optim.DefaultConfig(o.LR)
	cfg.WeightDecay = o.WeightDecay
	opt := optim.New(cfg)
	rng := rand.New(rand.NewSource(o.Seed))
	det := NewDetector(m)
	best := DetectionScore{}
	bestF1 := -1.0
	type crop struct {
		ink   []float32
		boxes []Box
	}
	batch := make([]crop, o.Batch)
	var logLoss float64
	start := time.Now()
	for step := 1; step <= o.Steps; step++ {
		if err := ctx.Err(); err != nil {
			return best, err
		}
		// Crops drawn in order (one generator), computed in parallel.
		seeds := make([]int64, o.Batch)
		pick := make([]int, o.Batch)
		for i := range batch {
			pick[i], seeds[i] = rng.Intn(len(cache)), rng.Int63()
		}
		linalg.Parallel(o.Batch, 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				ink, boxes := cache[pick[i]].crop(rand.New(rand.NewSource(seeds[i])), o)
				batch[i] = crop{ink, boxes}
			}
		})
		scale := float32(1 / float64(o.Batch))
		linalg.Parallel(len(slots), 1, func(lo, hi int) {
			for k := lo; k < hi; k++ {
				sl := slots[k]
				sl.g.Zero()
				sl.loss = 0
				for i := k; i < len(batch); i += len(slots) {
					out := m.Forward(&sl.t, batch[i].ink, o.Crop, o.Crop, true)
					sl.d = grow(sl.d, len(out.V))
					sl.loss += detLoss(m, out.V, sl.d, out.S.H, out.S.W, batch[i].boxes, scale)
					m.Backward(&sl.t, out, sl.d, sl.g)
				}
			}
		})
		total.Zero()
		for _, sl := range slots {
			total.Add(sl.g)
			logLoss += sl.loss
		}
		norm := optim.ClipGradNorm(o.Clip, dense, nil)
		lr := schedule(TrainOptions{LR: o.LR, Warmup: o.Warmup, Steps: o.Steps}, step)
		opt.Step(lr, dense, nil)

		if o.LogEvery > 0 && step%o.LogEvery == 0 {
			fmt.Fprintf(o.Log, "step %6d  loss %7.4f  |g| %6.2f  lr %.2e  %5.1f crops/s\n",
				step, logLoss/float64(o.LogEvery*o.Batch), norm, lr, float64(step*o.Batch)/time.Since(start).Seconds())
			logLoss = 0
		}
		last := step == o.Steps
		if (len(o.Eval) > 0 && o.EvalEvery > 0 && step%o.EvalEvery == 0) || last {
			s := best
			if len(o.Eval) > 0 {
				var err error
				if s, err = EvaluateDetector(det, o.Eval, 0.5); err != nil {
					return best, err
				}
				fmt.Fprintf(o.Log, "step %6d  eval P %.3f  R %.3f  F1 %.3f\n", step, s.Precision(), s.Recall(), s.F1())
			}
			if s.F1() > bestF1 || len(o.Eval) == 0 {
				best, bestF1 = s, s.F1()
				if o.Out != "" {
					if err := SaveDetector(o.Out, m); err != nil {
						return best, err
					}
				}
			}
		}
	}
	return best, nil
}

func cachePage(g *image.Gray, p PageSample, longSide int) cachedPage {
	b := g.Bounds()
	sw, sh := b.Dx(), b.Dy()
	s := math.Min(1, float64(longSide)/float64(max(sw, sh)))
	dw, dh := max(1, int(math.Round(float64(sw)*s))), max(1, int(math.Round(float64(sh)*s)))
	src := make([]uint8, sw*sh)
	for y := range sh {
		copy(src[y*sw:(y+1)*sw], g.Pix[g.PixOffset(b.Min.X, b.Min.Y+y):])
	}
	pix := src
	if s < 1 {
		r := resizeArea(src, sw, sh, dw, dh)
		pix = make([]uint8, len(r))
		for i, v := range r {
			pix[i] = uint8(min(max(v+0.5, 0), 255))
		}
	}
	c := cachedPage{pix: pix, W: dw, H: dh}
	for _, bx := range p.Boxes {
		c.boxes = append(c.boxes, Box{bx[0] * s, bx[1] * s, bx[2] * s, bx[3] * s})
	}
	return c
}

// crop draws a training crop: the page at a random scale, a random window
// of o.Crop pixels (paper beyond the page), photometric damage.
func (c cachedPage) crop(r *rand.Rand, o DetTrainOptions) ([]float32, []Box) {
	long := o.MinSide + r.Intn(o.MaxSide-o.MinSide+1)
	s := float64(long) / float64(max(c.W, c.H))
	dw, dh := max(1, int(float64(c.W)*s)), max(1, int(float64(c.H)*s))
	var px []float32
	if s < 1 {
		px = resizeArea(c.pix, c.W, c.H, dw, dh)
	} else {
		px = resizeLinear(c.pix, c.W, c.H, dw, dh)
	}
	n := o.Crop
	ox := r.Intn(max(1, dw-n/2)) - n/4 // windows may stick out of the page a little
	oy := r.Intn(max(1, dh-n/2)) - n/4
	contrast := float32(0.6 + 0.6*r.Float64())
	shift := float32(0.1 * (2*r.Float64() - 1))
	noise := 0.05 * r.Float64()
	ink := make([]float32, n*n)
	for y := range n {
		sy := y + oy
		for x := range n {
			sx := x + ox
			v := float32(0)
			if sx >= 0 && sy >= 0 && sx < dw && sy < dh {
				v = 1 - px[sy*dw+sx]/255
			}
			v = v*contrast + shift + float32(r.NormFloat64()*noise)
			ink[y*n+x] = min(max(v, 0), 1)
		}
	}
	var boxes []Box
	for _, b := range c.boxes {
		bb := Box{b.X0*s - float64(ox), b.Y0*s - float64(oy), b.X1*s - float64(ox), b.Y1*s - float64(oy)}
		if bb.X1 > 0 && bb.Y1 > 0 && bb.X0 < float64(n) && bb.Y0 < float64(n) {
			boxes = append(boxes, bb)
		}
	}
	return ink, boxes
}

// detLoss computes the detector's loss on one crop and writes its
// gradient with respect to the output maps into d (scaled by scale):
// binary cross-entropy of the core map over the whole crop, and on the
// cores L1 on the log of the distances to the line's top and bottom.
func detLoss(m *linedet.Model, out, d []float32, H, W int, boxes []Box, scale float32) float64 {
	linalg.ZeroUpper()
	core := make([]bool, H*W)
	dist := make([]float32, 2*H*W)
	// Larger lines first: a smaller line drawn after keeps its own band.
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return boxes[order[a]].Y1-boxes[order[a]].Y0 > boxes[order[b]].Y1-boxes[order[b]].Y0
	})
	for _, k := range order {
		b := boxes[k]
		// Crop pixels to output pixels (stride 2).
		x0, y0, x1, y1 := b.X0/2, b.Y0/2, b.X1/2, b.Y1/2
		lh := y1 - y0
		if lh < 1 || x1-x0 < 1 {
			continue
		}
		cy, half := (y0+y1)/2, coreHeight*lh/2
		r0 := max(0, int(math.Floor(cy-half)))
		r1 := min(H, max(r0+1, int(math.Ceil(cy+half))))
		sx := min(coreShrinkX*lh, 0.25*(x1-x0))
		c0 := max(0, int(math.Floor(x0+sx)))
		c1 := min(W, max(c0+1, int(math.Ceil(x1-sx))))
		for y := r0; y < r1; y++ {
			yc := float32(y) + 0.5
			for x := c0; x < c1; x++ {
				core[y*W+x] = true
				dist[2*(y*W+x)] = yc - float32(y0)
				dist[2*(y*W+x)+1] = float32(y1) - yc
			}
		}
	}
	clear(d[:H*W*3])
	ncore := 0
	for _, c := range core {
		if c {
			ncore++
		}
	}
	npix := float64(H * W)
	var loss float64
	for i, c := range core {
		z := float64(out[i*3])
		y := 0.0
		if c {
			y = 1
		}
		loss += (math.Max(z, 0) - z*y + math.Log1p(math.Exp(-math.Abs(z)))) / npix
		d[i*3] = float32((1/(1+math.Exp(-z))-y)/npix) * scale
		if !c {
			continue
		}
		for k := range 2 {
			raw := float64(out[i*3+1+k])
			sp := raw
			if raw <= 20 {
				sp = math.Log1p(math.Exp(raw))
			}
			pred := sp * float64(m.Cfg.DistScale)
			diff := math.Log(max(pred, 1e-3)) - math.Log(max(float64(dist[2*i+k]), 0.25))
			loss += math.Abs(diff) / float64(2*ncore)
			if pred < 1e-3 {
				continue // the loss is clamped flat there: no gradient
			}
			sign := 1.0
			if diff < 0 {
				sign = -1
			}
			// d|log(pred) - log(t)|/draw = sign · σ(raw) / softplus(raw).
			d[i*3+1+k] = float32(sign/(1+math.Exp(-raw))/sp/float64(2*ncore)) * scale
		}
	}
	return loss
}

// EvaluateDetector detects the lines of the pages and matches them with
// the references at minIoU.
func EvaluateDetector(d *Detector, pages []PageSample, minIoU float64) (DetectionScore, error) {
	scores := make([]DetectionScore, len(pages))
	errs := make([]error, len(pages))
	linalg.Parallel(len(pages), 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			g, err := ReadImage(pages[i].Image)
			if err != nil {
				errs[i] = err
				continue
			}
			ref := make([]Box, len(pages[i].Boxes))
			for j, b := range pages[i].Boxes {
				ref[j] = Box{b[0], b[1], b[2], b[3]}
			}
			pred := d.Detect(g)
			s := DetectionScore{Ref: len(ref), Pred: len(pred)}
			for _, j := range Match(ref, pred, minIoU) {
				if j >= 0 {
					s.Matched++
				}
			}
			scores[i] = s
		}
	})
	var s DetectionScore
	for i, x := range scores {
		if errs[i] != nil {
			return s, errs[i]
		}
		s.Ref += x.Ref
		s.Pred += x.Pred
		s.Matched += x.Matched
	}
	return s, nil
}

// SaveDetector writes a detector to dir: config.json and
// model.safetensors.
func SaveDetector(dir string, m *linedet.Model) error {
	return saveFiles(dir, m.Cfg, m.Tensors())
}

// EvaluatePages reads the pages end to end, detection then recognition,
// and measures the error rates of the reference lines the recognizer's
// vocabulary can write: a reference line reads as its matched box's text,
// or as nothing when no box matches it.
func EvaluatePages(d *Detector, r *Recognizer, pages []PageSample, minIoU float64) (ErrorRate, error) {
	return EvaluatePagesWith(func(g *image.Gray, _ int) []Box { return d.Detect(g) }, r, pages, minIoU)
}

// EvaluatePagesWith is EvaluatePages with the boxes of page i given by
// detect (another detector's, for a comparison).
func EvaluatePagesWith(detect func(page *image.Gray, i int) []Box, r *Recognizer, pages []PageSample, minIoU float64) (ErrorRate, error) {
	enc := r.Vocab().Encoder()
	rates := make([]ErrorRate, len(pages))
	errs := make([]error, len(pages))
	linalg.Parallel(len(pages), 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			g, err := ReadImage(pages[i].Image)
			if err != nil {
				errs[i] = err
				continue
			}
			ref := make([]Box, len(pages[i].Boxes))
			for j, b := range pages[i].Boxes {
				ref[j] = Box{b[0], b[1], b[2], b[3]}
			}
			pred := detect(g, i)
			for j, k := range Match(ref, pred, minIoU) {
				want := NormalizeText(pages[i].Texts[j])
				if _, _, ok := enc.Encode(want); !ok {
					continue
				}
				got := ""
				if k >= 0 {
					got = r.ReadLine(g, pred[k]).Text
				}
				rates[i].Add(got, want)
			}
		}
	})
	var e ErrorRate
	for i, x := range rates {
		if errs[i] != nil {
			return e, errs[i]
		}
		e.Merge(x)
	}
	return e, nil
}
