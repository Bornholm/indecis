// Command ocrpdf turns born-digital PDFs into labeled OCR lines: each page
// rendered by poppler (pdftoppm), each line of its text layer
// (pdftotext -bbox-layout) as a sample: the page image, the line's ink box,
// its text. The text layer is the ground truth, without any reading.
//
//	go run ./ocrpdf -pdfs ~/.cache/indecis/datasets/ocr/hal/fr -out hal-fr -lang fr
//
// Writes out/lines.jsonl (ocr.Sample) and out/pages/*.png.
package main

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/bornholm/indecis/ocr"
)

type bboxDoc struct {
	Pages []struct {
		Width  float64 `xml:"width,attr"`
		Height float64 `xml:"height,attr"`
		Lines  []struct {
			XMin  float64  `xml:"xMin,attr"`
			YMin  float64  `xml:"yMin,attr"`
			XMax  float64  `xml:"xMax,attr"`
			YMax  float64  `xml:"yMax,attr"`
			Words []string `xml:"word"`
		} `xml:"flow>block>line"`
	} `xml:"body>doc>page"`
}

func main() {
	pdfs := flag.String("pdfs", "", "directory of PDFs")
	out := flag.String("out", "", "output directory")
	lang := flag.String("lang", "", "meta lang of the lines")
	first := flag.Int("first", 2, "first page read (HAL adds a cover page)")
	pages := flag.Int("pages", 4, "pages read per document")
	seed := flag.Int64("seed", 1, "random seed (resolution per document)")
	flag.Parse()
	if *pdfs == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	v, _ := ocr.NewVocab(ocr.DefaultCharset)
	enc := v.Encoder()
	files, _ := filepath.Glob(filepath.Join(*pdfs, "*.pdf"))
	sort.Strings(files)
	os.MkdirAll(filepath.Join(*out, "pages"), 0o755)
	f, err := os.Create(filepath.Join(*out, "lines.jsonl"))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	jw := json.NewEncoder(f)
	jw.SetEscapeHTML(false)
	r := rand.New(rand.NewSource(*seed))
	total, skipped := 0, 0
	for _, pdf := range files {
		doc := strings.TrimSuffix(filepath.Base(pdf), ".pdf")
		dpi := []int{100, 150, 200, 300}[r.Intn(4)]
		n, s, err := convert(pdf, doc, *out, *lang, dpi, *first, *pages, enc, jw)
		if err != nil {
			log.Printf("%s: %v", doc, err)
			continue
		}
		total += n
		skipped += s
	}
	log.Printf("%d documents, %d lines, %d skipped", len(files), total, skipped)
}

func convert(pdf, doc, out, lang string, dpi, first, pages int, enc *ocr.Encoder, jw *json.Encoder) (int, int, error) {
	last := first + pages - 1
	html, err := exec.Command("pdftotext", "-bbox-layout", "-f", fmt.Sprint(first), "-l", fmt.Sprint(last), pdf, "-").Output()
	if err != nil {
		return 0, 0, err
	}
	var d bboxDoc
	dec := xml.NewDecoder(bytes.NewReader(html))
	dec.Strict = false
	dec.AutoClose = xml.HTMLAutoClose
	dec.Entity = xml.HTMLEntity
	if err := dec.Decode(&d); err != nil {
		return 0, 0, err
	}
	// A text layer whose fonts map ligatures to wrong characters ("ti" read
	// ";", "tt" read "H") is no ground truth: drop the document.
	words, odd := 0, 0
	for _, p := range d.Pages {
		for _, l := range p.Lines {
			for _, w := range l.Words {
				words++
				if brokenWord.MatchString(w) {
					odd++
				}
			}
		}
	}
	if odd > 2 && float64(odd) > 0.002*float64(words) {
		return 0, 0, fmt.Errorf("text layer looks broken (%d/%d odd words)", odd, words)
	}
	var samples []ocr.Sample
	n, skipped := 0, 0
	for pi, p := range d.Pages {
		if len(p.Lines) == 0 {
			continue
		}
		pn := first + pi
		base := filepath.Join(out, "pages", fmt.Sprintf("%s-p%d", doc, pn))
		if err := exec.Command("pdftoppm", "-r", fmt.Sprint(dpi), "-gray", "-png", "-singlefile",
			"-f", fmt.Sprint(pn), "-l", fmt.Sprint(pn), pdf, base).Run(); err != nil {
			return n, skipped, err
		}
		page, err := ocr.ReadImage(base + ".png")
		if err != nil {
			return n, skipped, err
		}
		scale := float64(dpi) / 72
		for _, l := range p.Lines {
			text := ocr.NormalizeText(strings.Join(l.Words, " "))
			box := ocr.Box{X0: l.XMin * scale, Y0: l.YMin * scale, X1: l.XMax * scale, Y1: l.YMax * scale}
			ink, ok := inkBox(page, box)
			_, _, known := enc.Encode(text)
			h := ink.Y1 - ink.Y0
			if text == "" || !known || !ok || h < 6 || ink.X1-ink.X0 < h {
				skipped++
				continue
			}
			samples = append(samples, ocr.Sample{
				Image: filepath.Join("pages", filepath.Base(base)+".png"),
				Box:   []float64{ink.X0, ink.Y0, ink.X1, ink.Y1},
				Text:  text,
				Meta:  map[string]string{"source": "hal", "lang": lang, "doc": doc, "dpi": fmt.Sprint(dpi)},
			})
			n++
		}
	}
	for _, s := range samples {
		jw.Encode(s)
	}
	return n, skipped, nil
}

// brokenWord matches a lowercase letter, a character that cannot be inside
// a word, a lowercase letter: "obten;on", "CeHe".
var brokenWord = regexp.MustCompile(`\p{Ll}([^\p{L}'’.\-/]|\p{Lu})\p{Ll}`)

// inkBox tightens a text-layer box (the font's box) to the dark pixels
// inside it, as a line detector would box the line.
func inkBox(page *image.Gray, b ocr.Box) (ocr.Box, bool) {
	r := image.Rect(int(math.Floor(b.X0)), int(math.Floor(b.Y0)), int(math.Ceil(b.X1)), int(math.Ceil(b.Y1))).Intersect(page.Rect)
	if r.Empty() {
		return b, false
	}
	x0, y0, x1, y1 := r.Max.X, r.Max.Y, r.Min.X, r.Min.Y
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if page.Pix[page.PixOffset(x, y)] < 128 {
				x0, x1 = min(x0, x), max(x1, x+1)
				y0, y1 = min(y0, y), max(y1, y+1)
			}
		}
	}
	if x1 <= x0 {
		return b, false
	}
	return ocr.Box{X0: float64(x0), Y0: float64(y0), X1: float64(x1), Y1: float64(y1)}, true
}
