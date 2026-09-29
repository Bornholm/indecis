package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/gif" // image formats read from the examples
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/bornholm/indecis/vision"
)

// imageExample is a line of an image dataset: an image file, relative to
// the JSONL file, and its labels.
type imageExample struct {
	Image  string         `json:"image"`
	Labels map[string]any `json:"labels"`
}

func runTrainVision(args []string) error {
	fs := flag.NewFlagSet("train-vision", flag.ExitOnError)
	backbone := fs.String("backbone", "", "SigLIP encoder (config.json, model.safetensors, tokenizer.json)")
	schemaPath := fs.String("schema", "", "schema: JSON list of questions (see the guide)")
	trainPath := fs.String("train", "", `training examples: JSONL of {"image": path, "labels": {...}}, paths relative to the file`)
	testPath := fs.String("test", "", "test examples (default: 10% of training)")
	out := fs.String("out", "", "directory of the trained model")
	def := vision.DefaultHeadTrainOptions()
	epochs := fs.Int("epochs", def.Epochs, "epochs")
	batch := fs.Int("batch", def.BatchSize, "examples per batch")
	lr := fs.Float64("lr", def.LR, "learning rate")
	k := fs.Int("k", 16, "values per patch read by the head")
	seed := fs.Int64("seed", def.Seed, "seed")
	workers := fs.Int("workers", 2, "images encoded at the same time (at most the performance cores)")
	int8 := fs.Bool("int8", true, "int8 encoder, as in serving")
	fs.Parse(args)
	if err := required(fs, "backbone", "schema", "train", "out"); err != nil {
		return err
	}
	sf, err := readSchema(*schemaPath)
	if err != nil {
		return err
	}
	var opts []vision.Option
	if *int8 {
		opts = append(opts, vision.WithInt8())
	}
	m, err := vision.Load(*backbone, opts...)
	if err != nil {
		return err
	}
	train, err := readImageExamples(*trainPath)
	if err != nil {
		return err
	}
	var test []imageExample
	if *testPath != "" {
		if test, err = readImageExamples(*testPath); err != nil {
			return err
		}
	} else {
		r := rand.New(rand.NewSource(*seed))
		r.Shuffle(len(train), func(i, j int) { train[i], train[j] = train[j], train[i] })
		cut := len(train) / 10
		train, test = train[cut:], train[:cut]
	}
	start := time.Now()
	trainH, err := encodeImages(m, train, *workers)
	if err != nil {
		return err
	}
	testH, err := encodeImages(m, test, *workers)
	if err != nil {
		return err
	}
	log.Printf("%d training and %d test images encoded in %s", len(trainH), len(testH), time.Since(start).Round(time.Second))
	T, H := m.PatchShape()
	head, err := vision.NewHead(sf.schema, T, H, *k, *seed)
	if err != nil {
		return err
	}
	to := vision.HeadTrainOptions{Epochs: *epochs, BatchSize: *batch, LR: *lr, WeightDecay: def.WeightDecay, Seed: *seed,
		Progress: func(epoch int, loss float64) { log.Printf("epoch %d/%d loss %.4f", epoch, *epochs, loss) }}
	start = time.Now()
	if err := head.Fit(trainH, to); err != nil {
		return err
	}
	log.Printf("head trained in %s", time.Since(start).Round(time.Second))
	if err := vision.SaveHead(*out, *backbone, head); err != nil {
		return err
	}
	log.Printf("model written to %s", *out)
	if len(testH) > 0 {
		acc := head.Evaluate(testH)
		for _, q := range sf.schema {
			if a, ok := acc[q.Name]; ok {
				fmt.Printf("%-12s n=%-6d acc=%.3f\n", q.Name, len(testH), a)
			}
		}
	}
	return nil
}

func readImageExamples(path string) ([]imageExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dir := filepath.Dir(path)
	var out []imageExample
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for line := 1; sc.Scan(); line++ {
		var e imageExample
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		if e.Image == "" {
			return nil, fmt.Errorf(`%s:%d: "image" missing`, path, line)
		}
		if !filepath.IsAbs(e.Image) {
			e.Image = filepath.Join(dir, e.Image)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

// encodeImages computes the patch features of each image, workers at a
// time; the order of the examples is kept.
func encodeImages(m *vision.Model, examples []imageExample, workers int) ([]vision.HeadExample, error) {
	out := make([]vision.HeadExample, len(examples))
	errs := make([]error, len(examples))
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < max(1, workers); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				out[i], errs[i] = encodeImage(m, examples[i])
			}
		}()
	}
	for i := range examples {
		if i > 0 && i%1000 == 0 {
			log.Printf("%d/%d images encoded", i, len(examples))
		}
		next <- i
	}
	close(next)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("%s: %w", examples[i].Image, err)
		}
	}
	return out, nil
}

func encodeImage(m *vision.Model, e imageExample) (vision.HeadExample, error) {
	f, err := os.Open(e.Image)
	if err != nil {
		return vision.HeadExample{}, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return vision.HeadExample{}, err
	}
	p, err := m.Patches(img)
	if err != nil {
		return vision.HeadExample{}, err
	}
	return vision.HeadExample{Patches: vision.ToBF16(p), Labels: e.Labels}, nil
}
