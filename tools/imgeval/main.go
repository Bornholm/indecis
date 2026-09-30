// Command imgeval measures zero-shot image classification with a SigLIP
// model on Imagenette: each class is described by a caption, and an image
// gets the class whose caption scores highest.
//
//	go run ./tools/imgeval -data ~/.cache/indecis/datasets/imagenette/imagenette2-160/val -n 50
package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/bornholm/indecis/internal/siglip"
	"github.com/bornholm/indecis/tokenizer"
)

// Imagenette's ten classes: WordNet id, English and French names.
var classes = []struct{ id, en, fr string }{
	{"n01440764", "tench", "tanche"},
	{"n02102040", "English springer spaniel", "springer anglais"},
	{"n02979186", "cassette player", "lecteur de cassettes"},
	{"n03000684", "chainsaw", "tronçonneuse"},
	{"n03028079", "church", "église"},
	{"n03394916", "French horn", "cor d'harmonie"},
	{"n03417042", "garbage truck", "camion poubelle"},
	{"n03425413", "gas pump", "pompe à essence"},
	{"n03445777", "golf ball", "balle de golf"},
	{"n03888257", "parachute", "parachute"},
}

func main() {
	home, _ := os.UserHomeDir()
	modelDir := flag.String("model", filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256"), "SigLIP model directory")
	data := flag.String("data", "", "Imagenette split directory (one folder per WordNet id)")
	n := flag.Int("n", 50, "images per class")
	int8 := flag.Bool("int8", false, "int8 products")
	flag.Parse()

	m, err := siglip.Load(*modelDir, *int8)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(*modelDir, "tokenizer.json"))
	if err != nil {
		log.Fatal(err)
	}
	embedText := func(s string) []float32 {
		tt, err := m.Text()
		if err != nil {
			log.Fatal(err)
		}
		e, err := tt.Embed(tt.Pad(tok.Encode(s), tok.EosID(), tok.PadID()))
		if err != nil {
			log.Fatal(err)
		}
		return normalize(e)
	}
	prompts := map[string]func(i int) string{
		"en":       func(i int) string { return "a photo of a " + classes[i].en + "." },
		"en-lower": func(i int) string { return strings.ToLower("a photo of a " + classes[i].en + ".") },
		"fr":       func(i int) string { return "une photo de " + classes[i].fr + "." },
		"name":     func(i int) string { return classes[i].en },
	}
	names := []string{"en", "en-lower", "fr", "name"}
	texts := map[string][][]float32{}
	for _, p := range names {
		for i := range classes {
			texts[p] = append(texts[p], embedText(prompts[p](i)))
		}
	}

	correct := map[string]int{}
	total := 0
	var elapsed time.Duration
	for ci, c := range classes {
		files, _ := filepath.Glob(filepath.Join(*data, c.id, "*"))
		sort.Strings(files)
		for _, path := range files[:min(*n, len(files))] {
			img, err := decode(path)
			if err != nil {
				log.Fatal(err)
			}
			start := time.Now()
			px, err := m.Cfg.Preprocess(img)
			if err != nil {
				log.Fatal(err)
			}
			e, err := m.Vision.Embed(px)
			if err != nil {
				log.Fatal(err)
			}
			elapsed += time.Since(start)
			e = normalize(e)
			total++
			for _, p := range names {
				best, bestScore := 0, float32(math.Inf(-1))
				for j, t := range texts[p] {
					if s := dot(e, t); s > bestScore {
						best, bestScore = j, s
					}
				}
				if best == ci {
					correct[p]++
				}
			}
		}
	}
	if total == 0 {
		log.Fatal("no image found")
	}
	fmt.Printf("%d images, int8=%v, %.1f ms per image\n", total, *int8, float64(elapsed.Milliseconds())/float64(total))
	for _, p := range names {
		fmt.Printf("  %-9s accuracy %.1f%%  (e.g. %q)\n", p, 100*float64(correct[p])/float64(total), prompts[p](0))
	}
}

func decode(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
