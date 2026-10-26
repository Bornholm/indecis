package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/internal/linerec"
	"github.com/bornholm/indecis/ocr"
)

func runOCR(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: indecis ocr read|eval|train|shard [options] (indecis ocr <command> -h)")
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
	}
	return fmt.Errorf("unknown ocr command %q (read, eval, train, shard)", args[0])
}

func runOCRRead(args []string) error {
	fs := flag.NewFlagSet("ocr read", flag.ExitOnError)
	model := fs.String("model", "", "recognizer (config.json, model.safetensors, vocab.json)")
	boxFlag := fs.String("box", "", "line box x0,y0,x1,y1 in pixels (default: the image is one line)")
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
	r, err := ocr.Load(*model)
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
	fs.Parse(args)
	if err := required(fs, "model", "data"); err != nil {
		return err
	}
	r, err := ocr.Load(*model)
	if err != nil {
		return err
	}
	if filepath.Ext(*data) != ".jsonl" {
		lines, err := ocr.LoadLines(*data, r.CropOptions())
		if err != nil {
			return err
		}
		e := ocr.Evaluate(r.Model(), r.Vocab(), lines)
		printRate("all", e)
		return nil
	}
	samples, err := ocr.ReadSamples(*data)
	if err != nil {
		return err
	}
	got := make([]string, len(samples))
	errs := make([]error, len(samples))
	linalg.Parallel(len(samples), 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			page, err := ocr.ReadImage(samples[i].Image)
			if err != nil {
				errs[i] = err
				continue
			}
			got[i] = r.ReadLine(page, samples[i].BoxOf(page)).Text
		}
	})
	all := ocr.ErrorRate{}
	groups := map[string]*ocr.ErrorRate{}
	var wrong []map[string]any
	for i, s := range samples {
		if errs[i] != nil {
			return errs[i]
		}
		s.Text = ocr.NormalizeText(s.Text)
		if *fold {
			got[i], s.Text = strings.ToUpper(got[i]), strings.ToUpper(s.Text)
		}
		var e ocr.ErrorRate
		e.Add(got[i], s.Text)
		all.Merge(e)
		if *by != "" {
			k := s.Meta[*by]
			if groups[k] == nil {
				groups[k] = &ocr.ErrorRate{}
			}
			groups[k].Merge(e)
		}
		if got[i] != s.Text {
			wrong = append(wrong, map[string]any{"image": s.Image, "box": s.Box, "want": s.Text, "got": got[i]})
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
	val := fs.String("val", "", "validation lines (shard, or JSONL of samples)")
	out := fs.String("out", "", "output directory (best model on -val)")
	init := fs.String("init", "", "start from this recognizer instead of fresh weights")
	charset := fs.String("charset", "", "file holding the characters of a fresh model (default: French and English)")
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
	var paths []string
	for _, g := range strings.Split(*train, ",") {
		files, err := filepath.Glob(expandHome(strings.TrimSpace(g)))
		if err != nil {
			return err
		}
		paths = append(paths, files...)
	}
	sort.Strings(paths)
	src, err := ocr.NewShardSource(paths, *buffer, *seed)
	if err != nil {
		return err
	}

	var m *linerec.Model
	var v ocr.Vocab
	if *init != "" {
		r, err := ocr.Load(*init)
		if err != nil {
			return err
		}
		m, v = r.Model(), r.Vocab()
	} else {
		cs := ocr.DefaultCharset
		if *charset != "" {
			b, err := os.ReadFile(*charset)
			if err != nil {
				return err
			}
			cs = strings.TrimRight(string(b), "\n")
		}
		if v, err = ocr.NewVocab(cs); err != nil {
			return err
		}
		if m, err = linerec.New(linerec.Base(len(v.Chars)+1), *seed); err != nil {
			return err
		}
	}
	var evalLines []ocr.ShardLine
	if *val != "" {
		crop := ocr.DefaultCrop()
		crop.Height, crop.Window = m.Cfg.Height, m.Cfg.Window
		if evalLines, err = ocr.LoadLines(*val, crop); err != nil {
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
	lines, err := ocr.LoadLines(*data, ocr.DefaultCrop())
	if err != nil {
		return err
	}
	w, err := ocr.CreateShard(*out)
	if err != nil {
		return err
	}
	for _, l := range lines {
		if err := w.Write(l.Text, ocr.Crop{Ink: l.Ink, W: l.W}, l.Meta); err != nil {
			w.Close()
			return err
		}
	}
	log.Printf("%d lines", len(lines))
	return w.Close()
}
