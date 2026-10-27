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

Character error rate (CER), greedy decoding, against Indra's recognizer (v1.1.0) read by the same Go code on the same crops. Dot leaders ("......" in a table of contents) are cut to "..." in both texts before comparing (`ocr.CompareForm`): no reader counts them right.

| Set | Lines | v1 | v2 | v3 | Indra |
| --- | --- | --- | --- | --- | --- |
| Synthetic, held out | 2,000 | 2.90% | **2.58%** | 2.69% | 4.98% |
| HAL born-digital PDFs, French (documents not trained on) | 3,270 | 1.01% | 0.91% | 0.75% | **0.60%** |
| HAL born-digital PDFs, English | 3,543 | 0.55% | 0.51% | **0.36%** | 0.38% |
| CORD receipts, photographed (validation split) | 1,413 | | 14.79% | **4.43%** | 5.86% |
| SROIE receipts, scanned (in capitals) | 2,000 | 7.11% | 6.50% | 4.44% | **3.04%** |

Real lines did most: v3 reads English PDFs as well as Indra and receipts photographed like CORD's better (it saw 200 of CORD's training receipts); French PDFs and scanned receipts are still a little behind. The synthetic set is ours: Indra's numbers there say little.

- v1: Indra's architecture with a layer norm and 206 symbols (1.06 M parameters), 20,000 steps of 64 lines (1.28 M lines: 1 M synthetic lines, 25,000 lines of HAL PDFs), 80 minutes on the 14 cores of a Core Ultra 7 265U.
- v2: v1 trained 15,000 more steps on v1's lines and 600,000 new ones (receipt lines, typographic quotes, columns of wide spaces, dot-matrix print).
- v3: v2 trained 15,000 more steps, 70% synthetic lines, 20% lines of HAL PDFs (85,000), 10% lines of CORD receipts (3,181).

int8 changes no rate by more than 0.02 point.

Reading a line 800 pixels wide on one core (P-core), decoding included: 6.8 ms in float32, **4.6 ms in int8**. Indra's recognizer (1,698 symbols) takes 8.2 ms in float32 and 5.4 ms in int8 in Go, 5.4 ms in onnxruntime (float32, one thread).

### Pages

The detector follows Indra's `det.onnx` (same 90,568 parameters): a stride-2 stem, four stages of MobileNet blocks, an FPN merged at stride 4, and three maps at stride 2, the line cores and, from each core pixel, the distances to the line's top and bottom. Pages are read with their longer side at 1,024 pixels, again larger when the lines come out small.

Lines found at IoU 0.5 (F1), and end-to-end CER of the reference lines (a missed line counts as read empty), on the HAL evaluation pages and CORD's validation receipts, against Indra's detector and recognizer:

| Pages | indecis F1 | Indra F1 | indecis end to end | Indra end to end |
| --- | --- | --- | --- | --- |
| HAL, French (80 pages) | 0.893 | **0.908** | 2.28% | **1.24%** |
| HAL, English (73 pages) | **0.907** | 0.906 | **1.14%** | 1.20% |
| CORD receipts (100 photos) | **0.878** | 0.804 | **9.40%** | 10.11% |

Detector v5: 1,789 pages of HAL PDFs, 3,000 synthetic pages (columns, lists, tables), CORD's 200 training receipts. On French pages it still misses lines Indra finds (recall 0.896 against 0.939), mostly short text in table cells.

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
