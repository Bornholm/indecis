package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/bornholm/indecis/internal/linalg"
	"github.com/bornholm/indecis/ocr"
)

func runOCR(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: indecis ocr read|eval [options] (indecis ocr <command> -h)")
	}
	switch args[0] {
	case "read":
		return runOCRRead(args[1:])
	case "eval":
		return runOCREval(args[1:])
	}
	return fmt.Errorf("unknown ocr command %q (read, eval)", args[0])
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
	data := fs.String("data", "", "labeled lines (JSONL: image, box, text, meta)")
	by := fs.String("by", "", "also measure by value of this meta field")
	errorsOut := fs.String("errors", "", "write the misread lines here (JSONL)")
	fs.Parse(args)
	if err := required(fs, "model", "data"); err != nil {
		return err
	}
	r, err := ocr.Load(*model)
	if err != nil {
		return err
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
