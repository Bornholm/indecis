package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
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
	"strings"
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
	trainPath := fs.String("train", "", `training examples: JSONL of {"image": path, "labels": {...}}, paths relative to the file (comma-separated files)`)
	testPath := fs.String("test", "", "test examples (default: 10% of training)")
	out := fs.String("out", "", "directory of the trained model")
	def := vision.DefaultHeadTrainOptions()
	epochs := fs.Int("epochs", def.Epochs, "epochs")
	batch := fs.Int("batch", def.BatchSize, "examples per batch")
	lr := fs.Float64("lr", def.LR, "learning rate")
	k := fs.Int("k", 16, "values per patch read by the head")
	layer := fs.Int("layer", 0, "encoder layer the head reads (0: the final features, after the last LayerNorm; n: the hidden states after n layers); the encoder stops there when serving learned questions")
	seed := fs.Int64("seed", def.Seed, "seed")
	workers := fs.Int("workers", 2, "images encoded at the same time (at most the performance cores)")
	quantized := fs.Bool("int8", true, "int8 encoder, as in serving")
	cacheDir := fs.String("cache", "", "directory where the patch features of each image are kept, to train again without encoding")
	fs.Parse(args)
	if err := required(fs, "backbone", "schema", "train", "out"); err != nil {
		return err
	}
	sf, err := readSchema(*schemaPath)
	if err != nil {
		return err
	}
	if *k <= 0 {
		return fmt.Errorf("-k must be positive, got %d", *k)
	}
	if *epochs < 1 || *lr <= 0 {
		return fmt.Errorf("-epochs %d, -lr %g: at least one epoch and a positive rate", *epochs, *lr)
	}
	var opts []vision.Option
	if *quantized {
		opts = append(opts, vision.WithInt8())
	}
	m, err := vision.Load(*backbone, opts...)
	if err != nil {
		return err
	}
	if *layer < 0 || *layer > m.Layers() {
		return fmt.Errorf("-layer %d: the encoder has %d layers", *layer, m.Layers())
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
		if cut == 0 {
			log.Printf("%d training examples: none kept for testing, the accuracy is not measured", len(train))
		}
	}
	start := time.Now()
	// The key names the encoder's weights, not only their directory: a model
	// replaced in place must not reuse features of the old one.
	weights, err := os.Stat(filepath.Join(*backbone, "model.safetensors"))
	if err != nil {
		return err
	}
	enc := encoder{m: m, layer: *layer, cache: *cacheDir, warn: &sync.Once{},
		key: fmt.Sprintf("%s|%d|%d|int8=%v|layer=%d", filepath.Clean(*backbone), weights.Size(), weights.ModTime().UnixNano(), *quantized, *layer)}
	trainH, err := enc.images(train, *workers)
	if err != nil {
		return err
	}
	testH, err := enc.images(test, *workers)
	if err != nil {
		return err
	}
	log.Printf("%d training and %d test images encoded in %s", len(trainH), len(testH), time.Since(start).Round(time.Second))
	T, H := m.PatchShape()
	head, err := vision.NewHead(sf.schema, T, H, *k, *seed)
	if err != nil {
		return err
	}
	head.Layer = *layer
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
		acc, err := head.Evaluate(testH)
		if err != nil {
			return fmt.Errorf("test examples: %w", err)
		}
		for _, q := range sf.schema {
			if a, ok := acc[q.Name]; ok {
				fmt.Printf("%-12s n=%-6d acc=%.3f\n", q.Name, len(testH), a)
			}
		}
	}
	return nil
}

func readImageExamples(paths string) ([]imageExample, error) {
	var out []imageExample
	for _, path := range strings.Split(paths, ",") {
		ex, err := readImageFile(path)
		if err != nil {
			return nil, err
		}
		out = append(out, ex...)
	}
	return out, nil
}

func readImageFile(path string) ([]imageExample, error) {
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

// encoder computes patch features, reading and writing them in cache if
// set: one file per image, named after the image (path, size, date) and
// the encoder.
type encoder struct {
	m     *vision.Model
	layer int
	cache string
	key   string
	warn  *sync.Once // a failing cache is reported once
}

func (e encoder) cachePath(path string) (string, bool) {
	if e.cache == "" {
		return "", false
	}
	st, err := os.Stat(path)
	if err != nil {
		return "", false
	}
	abs, _ := filepath.Abs(path)
	h := sha256.Sum256([]byte(fmt.Sprintf("%s|%d|%d|%s", abs, st.Size(), st.ModTime().UnixNano(), e.key)))
	return filepath.Join(e.cache, hex.EncodeToString(h[:16])+".bf16"), true
}

// images computes the patch features of each image, workers at a time;
// the order of the examples is kept.
func (e encoder) images(examples []imageExample, workers int) ([]vision.HeadExample, error) {
	if e.cache != "" {
		if err := os.MkdirAll(e.cache, 0o755); err != nil {
			return nil, err
		}
	}
	out := make([]vision.HeadExample, len(examples))
	errs := make([]error, len(examples))
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < max(1, workers); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				out[i], errs[i] = e.image(examples[i])
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

func (e encoder) image(ex imageExample) (vision.HeadExample, error) {
	cp, cached := e.cachePath(ex.Image)
	if cached {
		if b, err := os.ReadFile(cp); err == nil {
			T, H := e.m.PatchShape()
			if len(b) == 2*T*H {
				p := make([]uint16, T*H)
				for i := range p {
					p[i] = binary.LittleEndian.Uint16(b[2*i:])
				}
				return vision.HeadExample{Patches: p, Labels: ex.Labels}, nil
			}
		}
	}
	h, err := encodeImage(e.m, ex, e.layer)
	if err == nil && cached {
		b := make([]byte, 2*len(h.Patches))
		for i, v := range h.Patches {
			binary.LittleEndian.PutUint16(b[2*i:], v)
		}
		// Written aside then renamed: a crash leaves no half file.
		tmp := cp + ".tmp"
		werr := os.WriteFile(tmp, b, 0o644)
		if werr == nil {
			werr = os.Rename(tmp, cp)
		}
		if werr != nil {
			os.Remove(tmp)
			e.warn.Do(func() { log.Printf("feature cache not written, images will be encoded again next time: %v", werr) })
		}
	}
	return h, err
}

func encodeImage(m *vision.Model, e imageExample, layer int) (vision.HeadExample, error) {
	f, err := os.Open(e.Image)
	if err != nil {
		return vision.HeadExample{}, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return vision.HeadExample{}, err
	}
	p, err := m.PatchesAt(img, layer)
	if err != nil {
		return vision.HeadExample{}, err
	}
	return vision.HeadExample{Patches: vision.ToBF16(p), Labels: e.Labels}, nil
}
