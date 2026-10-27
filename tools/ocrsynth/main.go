// Command ocrsynth generates synthetic text lines for training indecis's
// OCR line recognizer: running text and generated strings (amounts, dates,
// references), rendered with free fonts among neighbouring lines, damaged
// as scans and faxes are, then cut with ocr.CropLine as reading does. The
// lines are written as shards (see ocr.CreateShard).
//
//	go run . -fonts ~/.cache/indecis/fonts \
//	    -text fr.txt,en.txt -n 100000 -out ~/.cache/indecis/datasets/ocr/synth
//
// It is a separate module: rendering fonts needs golang.org/x/image, and
// the library depends on nothing outside the standard library.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/bornholm/indecis/ocr"
)

func main() {
	fontDirs := flag.String("fonts", "", "font directories (comma-separated), searched recursively")
	texts := flag.String("text", "", "text files, one paragraph per line (comma-separated)")
	maxParas := flag.Int("max-paragraphs", 200000, "paragraphs read per text file")
	n := flag.Int("n", 10000, "lines to generate")
	shardSize := flag.Int("shard-size", 5000, "lines per shard")
	out := flag.String("out", "", "output directory")
	prefix := flag.String("prefix", "synth", "shard name prefix")
	seed := flag.Int64("seed", 1, "random seed: shard i draws from seed+i")
	workers := flag.Int("workers", runtime.NumCPU(), "parallel workers")
	preview := flag.Int("preview", 0, "also write the first lines' patches and crops as PNG")
	charset := flag.String("charset", "", "file holding the vocabulary's characters (default: ocr.DefaultCharset)")
	pages := flag.Int("pages", 0, "generate this many pages (out/pages.jsonl, out/pages/*.png) for detection, instead of lines")
	flag.Parse()
	if *fontDirs == "" || *texts == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	cs := ocr.DefaultCharset
	if *charset != "" {
		b, err := os.ReadFile(*charset)
		if err != nil {
			log.Fatal(err)
		}
		cs = strings.TrimRight(string(b), "\n")
	}
	v, err := ocr.NewVocab(cs)
	if err != nil {
		log.Fatal(err)
	}
	runes := []rune(cs)
	fonts, err := loadFonts(split(*fontDirs), runes)
	if err != nil {
		log.Fatal(err)
	}
	var corpora []*corpus
	for _, p := range split(*texts) {
		c, err := loadCorpus([]string{p}, *maxParas)
		if err != nil {
			log.Fatal(err)
		}
		corpora = append(corpora, c)
	}
	set := newFontSet(fonts)
	log.Printf("%d fonts in %d families, %d corpora", len(fonts), len(set.families), len(corpora))
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	if *pages > 0 {
		writePages(fonts, corpora, runes, v, *pages, *out, *seed, *workers)
		return
	}
	shards := (*n + *shardSize - 1) / *shardSize
	jobs := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	start := time.Now()
	for range *workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g := &generator{fonts: set, text: &texter{corpora: corpora, charset: runes, enc: v.Encoder()}, faces: faces{}, crop: ocr.DefaultCrop()}
			for i := range jobs {
				count := min(*shardSize, *n-i**shardSize)
				path := filepath.Join(*out, fmt.Sprintf("%s-%d-%05d.lines", *prefix, *seed, i))
				if _, err := os.Stat(path); err == nil {
					continue // written by an earlier run: shard i always draws the same lines
				}
				if err := writeShard(g, path, count, *seed+int64(i), i == 0 && *preview > 0, *preview); err != nil {
					log.Fatal(err)
				}
				mu.Lock()
				done += count
				log.Printf("%s: %d lines (%d/%d, %.0f lines/s)", filepath.Base(path), count, done, *n, float64(done)/time.Since(start).Seconds())
				mu.Unlock()
			}
		}()
	}
	for i := range shards {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
}

func writeShard(g *generator, path string, n int, seed int64, preview bool, nPreview int) error {
	r := rand.New(rand.NewSource(seed))
	w, err := ocr.CreateShard(path + ".tmp")
	if err != nil {
		return err
	}
	for i := range n {
		s := g.sample(r)
		c := ocr.CropLine(s.page, s.box, g.crop)
		if err := w.Write(s.text, c, s.meta); err != nil {
			w.Close()
			return err
		}
		if preview && i < nPreview {
			dir := filepath.Join(filepath.Dir(path), "preview")
			os.MkdirAll(dir, 0o755)
			savePNG(filepath.Join(dir, fmt.Sprintf("%03d-page.png", i)), s.page)
			crop := image.NewGray(image.Rect(0, 0, c.W, len(c.Ink)/c.W))
			for j, v := range c.Ink {
				crop.Pix[j] = 255 - uint8(v*255+0.5)
			}
			savePNG(filepath.Join(dir, fmt.Sprintf("%03d-crop.png", i)), crop)
			os.WriteFile(filepath.Join(dir, fmt.Sprintf("%03d.txt", i)), []byte(s.text+"\n"+fmt.Sprint(s.meta)+"\n"), 0o644)
		}
	}
	if err := w.Close(); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

func savePNG(path string, img image.Image) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	png.Encode(f, img)
}

func split(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if strings.HasPrefix(p, "~/") {
			home, _ := os.UserHomeDir()
			p = filepath.Join(home, p[2:])
		}
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// writePages renders pages for detection training: page i draws from
// seed+i.
func writePages(fonts []*fontFile, corpora []*corpus, runes []rune, v ocr.Vocab, n int, out string, seed int64, workers int) {
	set := newFontSet(fonts)
	dir := filepath.Join(out, "pages")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Fatal(err)
	}
	f, err := os.Create(filepath.Join(out, "pages.jsonl"))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	jobs := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	start := time.Now()
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g := &generator{fonts: set, text: &texter{corpora: corpora, charset: runes, enc: v.Encoder()}, faces: faces{}, crop: ocr.DefaultCrop()}
			for i := range jobs {
				p := g.page(rand.New(rand.NewSource(seed + int64(i))))
				name := fmt.Sprintf("synth-%d-%06d.png", seed, i)
				savePNG(filepath.Join(dir, name), p.page)
				s := ocr.PageSample{Image: filepath.Join("pages", name), Texts: p.texts, Meta: map[string]string{"source": "synth"}}
				for _, b := range p.boxes {
					s.Boxes = append(s.Boxes, [4]float64{b.X0, b.Y0, b.X1, b.Y1})
				}
				mu.Lock()
				enc.Encode(s)
				if (i+1)%100 == 0 {
					log.Printf("%d pages (%.1f pages/s)", i+1, float64(i+1)/time.Since(start).Seconds())
				}
				mu.Unlock()
			}
		}()
	}
	for i := range n {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
}
