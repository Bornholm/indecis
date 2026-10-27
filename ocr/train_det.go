package ocr

import (
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand"
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
	if o.Crop <= 0 {
		o.Crop = 512
	}
	if o.MinSide <= 0 {
		o.MinSide, o.MaxSide = 640, 1280
	}
	if o.Log == nil {
		o.Log = io.Discard
	}
	o.Crop = o.Crop / linedet.Multiple * linedet.Multiple

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
// binary cross-entropy of the text map on line cores (the rest of each
// line ignored), L1 on the log of the line height on the cores. The
// picture map has no target yet.
func detLoss(m *linedet.Model, out, d []float32, H, W int, boxes []Box, scale float32) float64 {
	linalg.ZeroUpper()
	target := make([]int8, H*W) // 0 background, 1 core, -1 ignored
	height := make([]float32, H*W)
	for _, b := range boxes {
		// Crop pixels to output pixels (stride 2).
		x0, y0, x1, y1 := b.X0/2, b.Y0/2, b.X1/2, b.Y1/2
		h := y1 - y0
		for y := max(0, int(math.Floor(y0))); y < min(H, int(math.Ceil(y1))); y++ {
			for x := max(0, int(math.Floor(x0))); x < min(W, int(math.Ceil(x1))); x++ {
				if target[y*W+x] == 0 {
					target[y*W+x] = -1
				}
			}
		}
		yc := (y0 + y1) / 2
		cy0, cy1 := yc-coreHeight*h/2, yc+coreHeight*h/2
		cx0, cx1 := x0+coreMarginX*h, x1-coreMarginX*h
		if cx1 <= cx0 {
			cx0, cx1 = (x0+x1)/2-0.5, (x0+x1)/2+0.5
		}
		for y := max(0, int(math.Round(cy0))); y < min(H, max(int(math.Round(cy0))+1, int(math.Round(cy1)))); y++ {
			for x := max(0, int(math.Round(cx0))); x < min(W, int(math.Round(cx1))); x++ {
				target[y*W+x] = 1
				height[y*W+x] = float32(h)
			}
		}
	}
	clear(d[:H*W*3])
	var loss float64
	npix, ncore := 0, 0
	for _, t := range target {
		if t >= 0 {
			npix++
		}
		if t == 1 {
			ncore++
		}
	}
	for i, t := range target {
		if t < 0 {
			continue
		}
		z := float64(out[i*3])
		y := float64(t)
		w := 1.0
		if t == 1 {
			w = 2
		}
		// BCE with logits, stable.
		loss += w * (math.Max(z, 0) - z*y + math.Log1p(math.Exp(-math.Abs(z)))) / float64(npix)
		p := 1 / (1 + math.Exp(-z))
		d[i*3] = float32(w*(p-y)/float64(npix)) * scale
		if t == 1 {
			raw := float64(out[i*3+2])
			sp := math.Log1p(math.Exp(min(raw, 20))) // softplus
			if raw > 20 {
				sp = raw
			}
			pred := sp * float64(m.Cfg.HeightScale)
			diff := math.Log(max(pred, 1e-4)) - math.Log(float64(height[i]))
			loss += math.Abs(diff) / float64(ncore)
			sign := 1.0
			if diff < 0 {
				sign = -1
			}
			sig := 1 / (1 + math.Exp(-raw))
			d[i*3+2] = float32(sign*sig/max(sp, 1e-4)/float64(ncore)) * scale
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
			pred := d.Detect(g)
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
