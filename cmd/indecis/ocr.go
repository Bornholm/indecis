package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/bornholm/indecis/internal/charlm"
	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linedet"
	"github.com/bornholm/indecis/internal/linerec"
	"github.com/bornholm/indecis/internal/safetensors"
	"github.com/bornholm/indecis/ocr"
)

func runOCR(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: indecis ocr read|page|eval|train|shard|train-det|eval-det|lm [options] (indecis ocr <command> -h)")
	}
	switch args[0] {
	case "read":
		return runOCRRead(args[1:])
	case "eval":
		return runOCREval(args[1:])
	case "train":
		return runOCRTrain(args[1:])
	case "shard":
		return runOCRShard(args[1:])
	case "page":
		return runOCRPage(args[1:])
	case "train-det":
		return runOCRTrainDet(args[1:])
	case "eval-det":
		return runOCREvalDet(args[1:])
	case "lm":
		return runOCRLM(args[1:])
	}
	return fmt.Errorf("unknown ocr command %q (read, page, eval, train, shard, train-det, eval-det, lm)", args[0])
}

func runOCRRead(args []string) error {
	fs := flag.NewFlagSet("ocr read", flag.ExitOnError)
	model := fs.String("model", "", "recognizer (config.json, model.safetensors, vocab.json)")
	boxFlag := fs.String("box", "", "line box x0,y0,x1,y1 in pixels (default: the image is one line)")
	ro := recognizerFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: indecis ocr read -model dir [-box x0,y0,x1,y1] image...")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if err := required(fs, "model"); err != nil {
		return err
	}
	var box []float64
	if *boxFlag != "" {
		for _, f := range strings.Split(*boxFlag, ",") {
			v, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
			if err != nil {
				return fmt.Errorf("-box: %w", err)
			}
			box = append(box, v)
		}
		if len(box) != 4 {
			return fmt.Errorf("-box needs 4 values")
		}
	}
	r, err := ocr.Load(*model, ro.options()...)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	for _, path := range fs.Args() {
		page, err := ocr.ReadImage(path)
		if err != nil {
			return err
		}
		l := r.ReadLine(page, ocr.Sample{Box: box}.BoxOf(page))
		enc.Encode(map[string]any{"image": path, "text": l.Text, "confidence": l.Confidence})
	}
	return nil
}

func runOCREval(args []string) error {
	fs := flag.NewFlagSet("ocr eval", flag.ExitOnError)
	model := fs.String("model", "", "recognizer")
	data := fs.String("data", "", "labeled lines (JSONL: image, box, text, meta; or a shard)")
	by := fs.String("by", "", "also measure by value of this meta field")
	errorsOut := fs.String("errors", "", "write the misread lines here (JSONL)")
	fold := fs.Bool("fold-case", false, "compare in upper case (references transcribed in capitals, as SROIE's)")
	ro := recognizerFlags(fs)
	fs.Parse(args)
	if err := required(fs, "model", "data"); err != nil {
		return err
	}
	r, err := ocr.Load(*model, ro.options()...)
	if err != nil {
		return err
	}
	type item struct {
		image, text string
		box         []float64
		line        int // index in a shard (no image, no box)
		meta        map[string]string
		got         string
	}
	var items []item
	var firstErr error
	if filepath.Ext(*data) != ".jsonl" {
		lines, err := ocr.LoadLines(*data, r.CropOptions())
		if err != nil {
			return err
		}
		items = make([]item, len(lines))
		linalg.Parallel(len(lines), 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				l := lines[i]
				items[i] = item{image: *data, line: i, text: l.Text, meta: l.Meta, got: r.ReadCrop(ocr.Crop{Ink: l.Ink, W: l.W}).Text}
			}
		})
	} else {
		samples, err := ocr.ReadSamples(*data)
		if err != nil {
			return err
		}
		// One decode per page, its lines read together.
		var pages []string
		byPage := map[string][]int{}
		items = make([]item, len(samples))
		for i, s := range samples {
			if _, ok := byPage[s.Image]; !ok {
				pages = append(pages, s.Image)
			}
			byPage[s.Image] = append(byPage[s.Image], i)
			items[i] = item{image: s.Image, text: s.Text, box: s.Box, meta: s.Meta}
		}
		errs := make([]error, len(pages))
		linalg.Parallel(len(pages), 1, func(lo, hi int) {
			for k := lo; k < hi; k++ {
				page, err := ocr.ReadImage(pages[k])
				if err != nil {
					errs[k] = err
					continue
				}
				for _, i := range byPage[pages[k]] {
					items[i].got = r.ReadLine(page, samples[i].BoxOf(page)).Text
				}
			}
		})
		for _, err := range errs {
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	if firstErr != nil {
		return firstErr
	}
	all := ocr.ErrorRate{}
	groups := map[string]*ocr.ErrorRate{}
	var wrong []map[string]any
	for _, s := range items {
		got := s.got
		s.text = ocr.NormalizeText(s.text)
		if *fold {
			got, s.text = strings.ToUpper(got), strings.ToUpper(s.text)
		}
		var e ocr.ErrorRate
		e.Add(got, s.text)
		all.Merge(e)
		if *by != "" {
			k := s.meta[*by]
			if groups[k] == nil {
				groups[k] = &ocr.ErrorRate{}
			}
			groups[k].Merge(e)
		}
		if ocr.CompareForm(got) != ocr.CompareForm(s.text) {
			w := map[string]any{"want": s.text, "got": got}
			if s.box != nil {
				w["image"], w["box"] = s.image, s.box
			} else {
				w["shard"], w["line"] = s.image, s.line
			}
			wrong = append(wrong, w)
		}
	}
	printRate("all", all)
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		printRate(*by+"="+k, *groups[k])
	}
	if *errorsOut != "" {
		f, err := os.Create(*errorsOut)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(f)
		enc.SetEscapeHTML(false)
		for _, w := range wrong {
			enc.Encode(w)
		}
		return f.Close()
	}
	return nil
}

func printRate(name string, e ocr.ErrorRate) {
	fmt.Printf("%-24s lines %6d  CER %6.2f%%  WER %6.2f%%  exact %5.1f%%\n",
		name, e.Lines, 100*e.CER(), 100*e.WER(), 100*float64(e.Exact)/float64(max(1, e.Lines)))
}

func runOCRTrain(args []string) error {
	fs := flag.NewFlagSet("ocr train", flag.ExitOnError)
	train := fs.String("train", "", "training shards (comma-separated patterns)")
	mix := fs.String("mix", "", "other shard sets drawn at a fixed share whatever their size: 'patterns=share;patterns=share' (the -train shards get the rest)")
	val := fs.String("val", "", "validation lines (shard, or JSONL of samples)")
	out := fs.String("out", "", "output directory (best model on -val)")
	init := fs.String("init", "", "start from this recognizer instead of fresh weights")
	charset := fs.String("charset", "", "file holding the characters of a fresh model (default: French and English)")
	extend := fs.Bool("extend", false, "with -init: extend the model's vocabulary to -charset (or the default charset), keeping the symbols it has")
	steps := fs.Int("steps", 20000, "optimizer steps")
	batch := fs.Int("batch", 64, "lines per step")
	lr := fs.Float64("lr", 2e-3, "peak learning rate")
	warmup := fs.Int("warmup", 1000, "warmup steps")
	wd := fs.Float64("wd", 0.01, "weight decay")
	clip := fs.Float64("clip", 5, "gradient norm clipping")
	buffer := fs.Int("buffer", 20000, "shuffle buffer, in lines")
	evalEvery := fs.Int("eval-every", 1000, "validation every n steps")
	seed := fs.Int64("seed", 1, "random seed")
	fs.Parse(args)
	if err := required(fs, "train", "out"); err != nil {
		return err
	}
	glob := func(patterns string) ([]string, error) {
		var paths []string
		for _, g := range strings.Split(patterns, ",") {
			files, err := filepath.Glob(expandHome(strings.TrimSpace(g)))
			if err != nil {
				return nil, err
			}
			paths = append(paths, files...)
		}
		sort.Strings(paths)
		return paths, nil
	}
	paths, err := glob(*train)
	if err != nil {
		return err
	}
	var src ocr.LineSource
	if src, err = ocr.NewShardSource(paths, *buffer, *seed); err != nil {
		return err
	}
	if *mix != "" {
		sources := []ocr.LineSource{src}
		weights := []float64{1}
		for i, entry := range strings.Split(*mix, ";") {
			pat, share, ok := strings.Cut(entry, "=")
			w, err := strconv.ParseFloat(share, 64)
			if !ok || err != nil || w <= 0 || w >= 1 {
				return fmt.Errorf("-mix: %q is not patterns=share", entry)
			}
			ps, err := glob(pat)
			if err != nil {
				return err
			}
			rs, err := ocr.NewShardSource(ps, *buffer, *seed+int64(i)+1)
			if err != nil {
				return err
			}
			sources = append(sources, rs)
			weights = append(weights, w)
			weights[0] -= w
			paths = append(paths, ps...)
		}
		if weights[0] <= 0 {
			return fmt.Errorf("-mix: the shares leave nothing to -train")
		}
		if src, err = ocr.NewMixSource(sources, weights, *seed+100); err != nil {
			return err
		}
	}

	var m *linerec.Model
	var v ocr.Vocab
	newVocab := func() (ocr.Vocab, error) {
		cs := ocr.DefaultCharset
		if *charset != "" {
			b, err := os.ReadFile(*charset)
			if err != nil {
				return ocr.Vocab{}, err
			}
			cs = strings.TrimRight(string(b), "\n")
		}
		return ocr.NewVocab(cs)
	}
	if *init != "" {
		r, err := ocr.Load(*init)
		if err != nil {
			return err
		}
		m, v = r.Model(), r.Vocab()
		if *extend {
			nv, err := newVocab()
			if err != nil {
				return err
			}
			t := m.Tensors()
			h, b := ocr.ExtendHead(t["head.weight"].Data, t["head.bias"].Data, m.Cfg.Dim, v, nv, *seed)
			cfg := m.Cfg
			cfg.Vocab = len(nv.Chars) + 1
			t["head.weight"] = safetensors.Tensor{Shape: []int{cfg.Vocab, cfg.Dim, 1}, Data: h}
			t["head.bias"] = safetensors.Tensor{Shape: []int{cfg.Vocab}, Data: b}
			if m, err = linerec.FromTensors(cfg, t); err != nil {
				return err
			}
			log.Printf("vocabulary extended from %d to %d characters", len(v.Chars), len(nv.Chars))
			v = nv
		}
	} else {
		if v, err = newVocab(); err != nil {
			return err
		}
		if m, err = linerec.New(linerec.Base(len(v.Chars)+1), *seed); err != nil {
			return err
		}
	}
	var evalLines []ocr.ShardLine
	if *val != "" {
		if evalLines, err = ocr.LoadLines(*val, ocr.ModelCrop(m)); err != nil {
			return err
		}
	}
	log.Printf("%d shards, %d validation lines, %d parameters", len(paths), len(evalLines), countParams(m))
	res, err := ocr.Train(context.Background(), m, v, src, ocr.TrainOptions{
		Steps: *steps, Batch: *batch, LR: *lr, Warmup: *warmup, WeightDecay: *wd, Clip: *clip,
		Eval: evalLines, EvalEvery: *evalEvery, Out: *out, LogEvery: 50, Log: os.Stderr,
	})
	if err != nil {
		return err
	}
	log.Printf("%d steps, %d lines, %d skipped; best CER %.2f%% at step %d",
		res.Steps, res.Lines, res.Skipped, 100*res.BestCER, res.BestStep)
	return nil
}

func countParams(m *linerec.Model) int {
	n := 0
	for _, p := range m.Params() {
		n += len(p.W)
	}
	return n
}

func expandHome(p string) string {
	if strings.HasPrefix(p, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, p[2:])
	}
	return p
}

func runOCRShard(args []string) error {
	fs := flag.NewFlagSet("ocr shard", flag.ExitOnError)
	data := fs.String("data", "", "labeled lines (JSONL of samples)")
	out := fs.String("out", "", "shard to write")
	fs.Parse(args)
	if err := required(fs, "data", "out"); err != nil {
		return err
	}
	w, err := ocr.CreateShard(*out)
	if err != nil {
		return err
	}
	n := 0
	err = ocr.EachLine(*data, ocr.DefaultCrop(), func(l ocr.ShardLine) error {
		n++
		return w.Write(l.Text, ocr.Crop{Ink: l.Ink, W: l.W}, l.Meta)
	})
	if err != nil {
		w.Close()
		return err
	}
	log.Printf("%d lines", n)
	return w.Close()
}

// recognizerOpts are the reading flags shared by the ocr commands.
type recognizerOpts struct {
	int8        *bool
	lm          *string
	alpha, beta *float64
	gate        *float64
	beam, top   *int
	lower       *bool
	retry       *float64
}

func recognizerFlags(fs *flag.FlagSet) *recognizerOpts {
	d := ocr.DefaultLM()
	return &recognizerOpts{
		int8:  fs.Bool("int8", true, "int8 computation if the processor allows it"),
		lm:    fs.String("lm", "", "character language model (indecis ocr lm) to decode uncertain lines"),
		alpha: fs.Float64("lm-alpha", d.Alpha, "weight of the language model"),
		beta:  fs.Float64("lm-beta", d.Beta, "bonus per character"),
		gate:  fs.Float64("lm-gate", float64(d.Gate), "lines read at least this surely skip the language model"),
		beam:  fs.Int("lm-beam", d.Beam, "prefixes kept per step"),
		top:   fs.Int("lm-top", d.Top, "characters tried per step"),
		lower: fs.Bool("lm-lower", d.LowerOnly, "the language model weighs small letters only"),
		retry: fs.Float64("retry", 0.5, "read again, from boxes a little moved, the lines read less surely than this (0: never)"),
	}
}

// options returns the recognizer options; it exits on an unreadable model.
func (o *recognizerOpts) options() []ocr.Option {
	var opts []ocr.Option
	if *o.int8 {
		opts = append(opts, ocr.WithInt8())
	}
	if *o.retry > 0 {
		opts = append(opts, ocr.WithRetry(float32(*o.retry)))
	}
	if *o.lm != "" {
		lm, err := charlm.Load(expandHome(*o.lm))
		if err != nil {
			log.Fatal(err)
		}
		lo := ocr.DefaultLM()
		lo.Alpha, lo.Beta, lo.Gate, lo.Beam, lo.Top, lo.LowerOnly = *o.alpha, *o.beta, float32(*o.gate), *o.beam, *o.top, *o.lower
		opts = append(opts, ocr.WithLanguageModel(lm, lo))
	}
	return opts
}

func runOCRLM(args []string) error {
	fs := flag.NewFlagSet("ocr lm", flag.ExitOnError)
	text := fs.String("text", "", "text files, one paragraph per line (comma-separated patterns)")
	out := fs.String("out", "", "model file")
	order := fs.Int("order", 6, "longest n-gram, in characters")
	minCount := fs.Uint("min", 2, "drop the n-grams seen fewer times")
	maxBytes := fs.Int64("max-bytes", 40<<20, "text read per file")
	fs.Parse(args)
	if err := required(fs, "text", "out"); err != nil {
		return err
	}
	b := charlm.NewBuilder(*order)
	for _, g := range strings.Split(*text, ",") {
		files, err := filepath.Glob(expandHome(strings.TrimSpace(g)))
		if err != nil {
			return err
		}
		for _, f := range files {
			fh, err := os.Open(f)
			if err != nil {
				return err
			}
			sc := bufio.NewScanner(fh)
			sc.Buffer(make([]byte, 1<<20), 1<<24)
			var n int64
			for sc.Scan() && n < *maxBytes {
				t := ocr.NormalizeText(sc.Text())
				n += int64(len(t))
				b.Add(t)
			}
			fh.Close()
			log.Printf("%s: %d MB", f, n>>20)
		}
	}
	m := b.Build(uint32(*minCount))
	ng, ctx := m.Size()
	log.Printf("%d n-grams, %d contexts", ng, ctx)
	return m.Save(*out)
}

func readPages(patterns string) ([]ocr.PageSample, error) {
	var out []ocr.PageSample
	for _, g := range strings.Split(patterns, ",") {
		files, err := filepath.Glob(expandHome(strings.TrimSpace(g)))
		if err != nil {
			return nil, err
		}
		if len(files) == 0 {
			return nil, fmt.Errorf("no file for %q", g)
		}
		sort.Strings(files)
		for _, f := range files {
			ps, err := ocr.ReadPageSamples(f)
			if err != nil {
				return nil, err
			}
			out = append(out, ps...)
		}
	}
	return out, nil
}

func runOCRPage(args []string) error {
	fs := flag.NewFlagSet("ocr page", flag.ExitOnError)
	det := fs.String("det", "", "line detector")
	model := fs.String("model", "", "recognizer")
	ro := recognizerFlags(fs)
	format := fs.String("format", "json", "json (lines with boxes), html (paragraphs) or text (paragraphs)")
	deskew := fs.Bool("deskew", true, "straighten slanted pages first (boxes then refer to the straightened page)")
	orient := fs.Bool("orient", true, "turn pages scanned sideways or upside down upright first (boxes then refer to the turned page)")
	fs.Parse(args)
	if err := required(fs, "det", "model"); err != nil {
		return err
	}
	d, err := ocr.LoadDetector(*det)
	if err != nil {
		return err
	}
	r, err := ocr.Load(*model, ro.options()...)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	for _, path := range fs.Args() {
		page, err := ocr.ReadImage(path)
		if err != nil {
			return err
		}
		var boxes []ocr.Box
		turns := 0
		if *orient {
			page, turns, boxes = ocr.Orient(d, r, page)
		}
		angle := 0.0
		if *deskew {
			page, angle = ocr.Deskew(page)
		}
		if boxes == nil || angle != 0 {
			boxes = d.Detect(page)
		}
		boxes, lines := ocr.Clean(boxes, r.ReadLines(page, boxes))
		type line struct {
			Box        [4]float64 `json:"box"`
			Text       string     `json:"text"`
			Confidence float32    `json:"confidence"`
		}
		switch *format {
		case "html":
			enc.Encode(map[string]any{"image": path, "html": ocr.PageHTML(ocr.Paragraphs(boxes, lines))})
			continue
		case "text":
			enc.Encode(map[string]any{"image": path, "text": strings.Join(ocr.Paragraphs(boxes, lines), "\n\n")})
			continue
		}
		out := make([]line, len(lines))
		for i, l := range lines {
			b := boxes[i]
			out[i] = line{[4]float64{b.X0, b.Y0, b.X1, b.Y1}, l.Text, l.Confidence}
		}
		enc.Encode(map[string]any{"image": path, "turns": turns, "angle": angle, "lines": out})
	}
	return nil
}

func runOCRTrainDet(args []string) error {
	fs := flag.NewFlagSet("ocr train-det", flag.ExitOnError)
	pages := fs.String("pages", "", "training pages (JSONL of ocr.PageSample, comma-separated patterns)")
	val := fs.String("val", "", "validation pages")
	out := fs.String("out", "", "output directory")
	init := fs.String("init", "", "start from this detector")
	steps := fs.Int("steps", 10000, "optimizer steps")
	batch := fs.Int("batch", 16, "crops per step")
	crop := fs.Int("crop", 512, "crop side in pixels")
	lr := fs.Float64("lr", 3e-3, "peak learning rate")
	warmup := fs.Int("warmup", 300, "warmup steps")
	evalEvery := fs.Int("eval-every", 1000, "validation every n steps")
	seed := fs.Int64("seed", 1, "random seed")
	fs.Parse(args)
	if err := required(fs, "pages", "out"); err != nil {
		return err
	}
	train, err := readPages(*pages)
	if err != nil {
		return err
	}
	var evalPages []ocr.PageSample
	if *val != "" {
		if evalPages, err = readPages(*val); err != nil {
			return err
		}
	}
	var m *linedet.Model
	if *init != "" {
		d, err := ocr.LoadDetector(*init)
		if err != nil {
			return err
		}
		m = d.Model()
	} else {
		m = linedet.New(linedet.Base(), *seed)
	}
	log.Printf("%d training pages, %d validation pages", len(train), len(evalPages))
	s, err := ocr.TrainDetector(context.Background(), m, train, ocr.DetTrainOptions{
		Steps: *steps, Batch: *batch, LR: *lr, Warmup: *warmup, WeightDecay: 0.01, Clip: 5, Crop: *crop,
		Seed: *seed, Eval: evalPages, EvalEvery: *evalEvery, Out: *out, LogEvery: 50, Log: os.Stderr,
	})
	if err != nil {
		return err
	}
	log.Printf("best: P %.3f R %.3f F1 %.3f", s.Precision(), s.Recall(), s.F1())
	return nil
}

func runOCREvalDet(args []string) error {
	fs := flag.NewFlagSet("ocr eval-det", flag.ExitOnError)
	det := fs.String("det", "", "line detector")
	model := fs.String("model", "", "recognizer: also measure the end-to-end CER of the reference lines")
	pages := fs.String("pages", "", "labeled pages (JSONL)")
	iou := fs.Float64("iou", 0.5, "minimum IoU of a match")
	ro := recognizerFlags(fs)
	boxesFile := fs.String("boxes", "", "instead of -det, the boxes of another detector: JSONL of {image, boxes}, image as in -pages")
	longSide := fs.Int("long-side", 0, "with -det: the length pages are read at (default: the detector's, 1024)")
	threshold := fs.Float64("threshold", 0, "with -det: the line probability threshold (default: the detector's, 0.5)")
	fs.Parse(args)
	if err := required(fs, "pages"); err != nil {
		return err
	}
	ps, err := readPages(*pages)
	if err != nil {
		return err
	}
	var detect func(*image.Gray, int) []ocr.Box
	switch {
	case *boxesFile != "":
		given, err := ocr.ReadPageSamples(*boxesFile)
		if err != nil {
			return err
		}
		// Pages are matched by file name: the two files may sit in
		// different directories. A name must then be unique, and every
		// page must have its boxes.
		byImage := map[string][]ocr.Box{}
		for _, g := range given {
			k := filepath.Base(g.Image)
			if _, dup := byImage[k]; dup {
				return fmt.Errorf("-boxes: two pages named %s", k)
			}
			bs := []ocr.Box{}
			for _, b := range g.Boxes {
				bs = append(bs, ocr.Box{X0: b[0], Y0: b[1], X1: b[2], Y1: b[3]})
			}
			byImage[k] = bs
		}
		seen := map[string]bool{}
		for _, p := range ps {
			k := filepath.Base(p.Image)
			if seen[k] {
				return fmt.Errorf("-pages: two pages named %s, -boxes cannot tell them apart", k)
			}
			seen[k] = true
			if _, ok := byImage[k]; !ok {
				return fmt.Errorf("-boxes has no entry for %s", k)
			}
		}
		detect = func(_ *image.Gray, i int) []ocr.Box { return byImage[filepath.Base(ps[i].Image)] }
	case *det != "":
		d, err := ocr.LoadDetector(*det)
		if err != nil {
			return err
		}
		if *longSide > 0 {
			d.LongSide = *longSide
		}
		if *threshold > 0 {
			d.Threshold = float32(*threshold)
		}
		detect = func(g *image.Gray, _ int) []ocr.Box { return d.Detect(g) }
	default:
		return fmt.Errorf("-det or -boxes is required")
	}
	var s ocr.DetectionScore
	for i, p := range ps {
		g, err := ocr.ReadImage(p.Image)
		if err != nil {
			return err
		}
		ref := make([]ocr.Box, len(p.Boxes))
		for j, b := range p.Boxes {
			ref[j] = ocr.Box{X0: b[0], Y0: b[1], X1: b[2], Y1: b[3]}
		}
		pred := detect(g, i)
		s.Ref += len(ref)
		s.Pred += len(pred)
		for _, k := range ocr.Match(ref, pred, *iou) {
			if k >= 0 {
				s.Matched++
			}
		}
	}
	fmt.Printf("detection  pages %d  lines %d  found %d  P %.3f  R %.3f  F1 %.3f\n", len(ps), s.Ref, s.Pred, s.Precision(), s.Recall(), s.F1())
	if *model == "" {
		return nil
	}
	r, err := ocr.Load(*model, ro.options()...)
	if err != nil {
		return err
	}
	e, err := ocr.EvaluatePagesWith(detect, r, ps, *iou)
	if err != nil {
		return err
	}
	printRate("end to end", e)
	return nil
}
