# Reading text lines (OCR)

indecis reads printed text from images with two small models that it trains and runs in Go, on CPU: a line detector (91 k parameters) finds the lines of a page, a recognizer (1 M parameters) reads each line.

The architecture is that of [Indra OCR](https://github.com/datalab-to/indra)'s recognizer, read from its ONNX graph: the line resized to 32 pixels high in a window of three line heights, one step every 4 pixels, a context branch over the window's rows, ten depthwise blocks, a CTC head. indecis trains its own weights: Indra's are under a license that forbids competing with Datalab's API, and are only used locally, as a reference.

## Reading

```bash
indecis ocr page -det det -model rec page.png               # a whole page: its lines, top to bottom
indecis ocr read -model rec line.png                        # an image holding one line
indecis ocr read -model rec -box 120,340,980,372 page.png   # a line of a page
```

```go
r, _ := ocr.Load(dir)
page := ocr.Gray(img)
line := r.ReadLine(page, ocr.Box{X0: 120, Y0: 340, X1: 980, Y1: 372})
line.Text       // the text
line.Chars      // each character with its x span on the page
line.Confidence // the lowest probability among the line's steps
```

`ReadLines` reads the lines of a page in parallel; `ocr.ReadPage(det, rec, page)` finds them first. The characters' positions on the page let a redaction hide a word read in an image.

## In numbers

Character error rate (CER) on lines cut the same way for both engines, against Indra's recognizer (v1.1.0) read by the same Go code. References and readings are normalized alike (`ocr.NormalizeText`: accents precomposed, "¬" read as a hyphen) and compared with dot leaders ("......") cut to "..." (`ocr.CompareForm`): no reader counts them right.

| Set | Lines | indecis v7 + language model | Indra |
| --- | --- | --- | --- |
| HAL born-digital PDFs, French (documents not trained on) | 3,270 | **0.58%** | 0.60% |
| HAL born-digital PDFs, English | 3,543 | **0.34%** | 0.38% |
| CORD receipts, photographed (validation split) | 1,413 | **1.77%** | 5.86% |
| Old French prints, scanned (HTR-United: Argus des brevets 1910, Exposition universelle 1878, NuBIS) | 6,260 | **2.58%** | 3.95% |
| HierText, photographed text (validation split) | 3,000 | **9.67%** | 11.95% |
| SROIE receipts, scanned (in capitals) | 2,000 | 3.04% | 3.04% |

v7 learned from CORD's training receipts, FONDUE's old printed catalogs and HierText's training photographs (other documents than those measured, but of the same kinds): its CORD, old-print and HierText numbers favour it.

How the recognizer got there (CER, greedy decoding but from v4 on, which use the language model; v6 and v7 its version 3, in more languages):

| Set | v1 | v2 | v3 | v4 | v5 | v6 | v7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HAL PDFs, French | 1.01% | 0.91% | 0.75% | 0.62% | 0.58% | 0.56% | 0.58% |
| HAL PDFs, English | 0.55% | 0.51% | 0.36% | 0.32% | 0.32% | 0.33% | 0.34% |
| CORD receipts | | 14.79% | 4.43% | 2.63% | 2.21% | 2.36% | 1.77% |
| Old French prints | | | | | 2.37% | 2.34% | 2.58% |
| HierText photographs | | | | | 14.05% | 13.70% | 9.67% |
| SROIE receipts | 7.11% | 6.50% | 4.44% | 3.94% | 3.44% | 3.49% | 3.04% |

- v1: Indra's architecture with a layer norm (1.06 M parameters), 20,000 steps of 64 lines (1 M synthetic lines, 25,000 lines of HAL PDFs), 80 minutes on the 14 cores of a Core Ultra 7 265U.
- v2: 15,000 more steps with 600,000 new synthetic lines (receipts, typographic quotes, wide spaces, dot-matrix print).
- v3: 15,000 more steps, 30% of real lines (HAL PDFs, 200 CORD receipts).
- v4: 15,000 more steps, all 800 CORD training receipts and 494 more English HAL documents.
- v5: 20,000 more steps, the vocabulary extended from 189 to 305 characters (superscripts and subscripts, bullets and boxes, arrows, Greek, Central European and Turkish letters), synthetic lines with old prints' long s and more typographic quotes, real lines of DocLayNet (176,000) and FONDUE's catalogs (37,000).
- v6: 20,000 more steps, the vocabulary extended to 680 characters (the Latin letters of 33 more languages, Vietnamese included, the Cyrillic and Greek alphabets, circled numbers, logic signs), 300,000 synthetic lines of Wikipedia in 39 more languages, 150,000 in 27 script and handwriting-style typefaces.
- v7: 10,000 more steps with 188,000 photographed lines of HierText.

int8 changes no rate by more than 0.02 point. Reading a line 800 pixels wide on one core (P-core), greedy decoding: 6.8 ms in float32, **4.6 ms in int8**; Indra's recognizer (1,698 symbols) takes 8.2 ms in float32 and 5.4 ms in int8 in Go, 5.4 ms in onnxruntime (float32, one thread).

### Pages

The detector follows Indra's `det.onnx` (same 90,568 parameters): a stride-2 stem, four stages of MobileNet blocks, an FPN merged at stride 4, and three maps at stride 2, the line cores and, from each core pixel, the distances to the line's top and bottom. Pages are read with their longer side at 1,024 pixels, again larger when the lines come out small. `indecis ocr page` first straightens a slanted page (`ocr.Deskew`, up to 8°: the angle at which the ink is most concentrated on rows, unless each half of the page is level on its own, as two columns set at different heights are); a line tilted by its height over its length would otherwise be cut by its box. On Omni Parse Bench's pages turned by 2°, it took the share of text tests passed from 33% to 56%. Lines come in reading order, by recursive XY-cut (columns before rows when a gutter is wider than the gaps between blocks, or, narrower, between two columns of text; short cells facing each other row by row, as a receipt's labels and amounts, read row by row), and `Paragraphs` groups them.

Lines found at IoU 0.5 (F1), and end-to-end CER of the reference lines (a missed line counts as read empty), against Indra's detector and recognizer (its decoding re-implemented over its ONNX file):

| Pages | indecis F1 | Indra F1 | indecis end to end | Indra end to end |
| --- | --- | --- | --- | --- |
| HAL PDFs, French (80 pages) | 0.854 | **0.908** | 1.78% | **1.24%** |
| HAL PDFs, English (73 pages) | 0.878 | **0.906** | **1.02%** | 1.20% |
| CORD receipts (100 photos) | **0.890** | 0.804 | **6.86%** | 10.11% |
| Old French prints (129 scans) | 0.896 | **0.900** | **8.12%** | 8.71% |

Detector v10 with recognizer v7 and the language model version 3. Detector v6 was trained in one run (15,000 steps of 12 crops, 3 hours): 2,505 pages of HAL PDFs, 2,077 of DocLayNet, 3,000 synthetic pages (columns, lists, tables), CORD's 800 training receipts. v10 went on from it for 8,000 steps on the same pages, with two changes. Crops are damaged as photocopies and old typewritten forms are, four in ten: strokes thinned by blotches, a blur, stains of the paper, specks, contrast down to a third. And the regions of a page whose ink no labeled line covers (pictures, charts, maps: a PDF's text layer holds none of their text) count for nothing in the loss (`PageSample.Ignore`, `tools/ocrsynth/ignore_unlabeled.py`): shown as paper, their text taught the detector to skip text in figures. Its F1 on these sets went down (it finds the figures' text, which their references leave out) and its end-to-end error on French PDFs too (2.07% to 1.78%); on Omni Parse Bench, it took the image pages from 56.8 to 58.6.

Training it on HierText's photographs (v7, v8) helped photographs (F1 0.564 to 0.669 on HierText's validation pages, Indra 0.726) but cost documents, and FONDUE's pages, whose lines are not all labeled, cost old prints: neither is in v10.

### Language model

Uncertain lines (one step below 0.9) can be decoded with a character language model: a CTC prefix beam search whose score adds 0.2·log P of each new character and 1 per character. The model is a 6-gram with Witten-Bell smoothing (`indecis ocr lm`), over Wikipedia and the email corpora, plus the texts of CORD's training receipts, repeated: domain text counts more than volume. It sees quotes, dashes, bullets and boxes, and digits (superscripts and subscripts included) as one class each, so that it has no say where only the glyph's shape decides: without that, it turned "•" into "*" and curly quotes into straight ones.

```bash
indecis ocr lm -text 'wiki_fr.txt,wiki_en.txt,receipts.txt' -order 6 -out lm.bin
indecis ocr page -det det -model rec -lm lm.bin page.png
```

Its weights were set on held-out sets (3,000 lines of other HAL PDFs, CORD's test split). With recognizer v3, it took CORD from 4.43% to 3.66% and SROIE from 4.44% to 3.66%, French PDFs from 0.75% to 0.65%, English ones barely (0.36% to 0.35%). It costs about 3 ms per uncertain line of 400 pixels on one core.

### Omni Parse Bench

[Omni Parse Bench](https://github.com/datalab-to/omni_parse_bench), the benchmark of Indra's README, scores a page's HTML with pass/fail tests: text present (a sentence, at most one or two characters off), in order, repeated; table cells; layout blocks. indecis writes paragraphs, no tables nor layout blocks, so only the text family applies (`tools/bench/opb_indecis.py`, graded by the benchmark's own `opb grade`; `tools/bench/opb_compare.py` sets it against the published engines, test by test).

Share of the text tests passed, on the 2,777 pages read (160 pages could not be rendered or fetched), test by test against the published runs. Indra's published row reads the text layer of the PDF pages (its auto mode): the fair comparison is on the image pages.

| Tests | Tests | indecis | Indra | Tesseract | Surya OCR 2 |
| --- | --- | --- | --- | --- | --- |
| Latin script, printed, no math, image pages | 1,322 | 58.6 | 63.5 | 19.4 | 79.0 |
| Latin script, printed, no math, all pages | 2,732 | 60.4 | 68.1 | 33.4 | 78.7 |
| Pages turned by a quarter, a half or three quarters | 183 | 76.5 | 79.2 | | |
| All text tests | 8,007 | 27.9 | 59.2 | 17.8 | 80.5 |

With recognizer v7, detector v10 and the language model version 3 (the turned pages measured with detector v6). indecis passes far more than Tesseract and stays behind Indra. The whole benchmark is out of its reach in places: math (3,271 tests, written as LaTeX), handwriting, scripts other than Latin, Greek and Cyrillic. On Latin printed pages, most failures are text partly found, lines cut short or broken into pieces by the detector on faint photocopies and typewritten forms; a quarter are a sentence one or two characters off.

`indecis ocr page` turns a page scanned sideways or upside down upright first (`ocr.Orient`): it reads the page's twelve widest lines as it is and, unless they read surely, turned by each quarter, and keeps the way whose characters the recognizer is surest of. An upright page costs the reading of those lines; a turned one, three more detections.

## Measuring

```bash
indecis ocr eval -model dir -data lines.jsonl -by source -errors errors.jsonl
```

A line set is a JSONL file (`image`, `box`, `text`, `meta`; without a box the image is the line) or a shard. `-fold-case` compares in capitals, for references transcribed in capitals.

## Training

```bash
indecis ocr train -train 'synth/*.lines,real/*.lines' -val val.lines -out model -steps 20000 -batch 64
```

Training reads shards: lines already cut, as the recognizer reads them. `tools/ocrsynth` renders synthetic lines (free fonts, French and English text, generated amounts, dates and references, damage of scans and faxes), `tools/ocrsynth/ocrpdf` cuts the lines of born-digital PDFs with their text layer as ground truth, and `indecis ocr shard` turns a JSONL line set into a shard. The sources and their licenses are in [ocr-data.md](ocr-data.md).

The gradients of a batch are summed over 16 fixed slices, in order: a run gives the same model whatever the number of cores.
