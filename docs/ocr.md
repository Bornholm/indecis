# Reading text lines (OCR)

indecis reads printed text lines from images with a small recognizer (1 M parameters) that it trains and runs in Go, on CPU. A line is given by its box on the page; finding the lines of a page is not done yet (see [ocr-plan.md](ocr-plan.md), P8).

The architecture is that of [Indra OCR](https://github.com/datalab-to/indra)'s recognizer, read from its ONNX graph: the line resized to 32 pixels high in a window of three line heights, one step every 4 pixels, a context branch over the window's rows, ten depthwise blocks, a CTC head. indecis trains its own weights: Indra's are under a license that forbids competing with Datalab's API, and are only used locally, as a reference.

## Reading

```bash
indecis ocr read -model dir line.png                       # an image holding one line
indecis ocr read -model dir -box 120,340,980,372 page.png  # a line of a page
```

```go
r, _ := ocr.Load(dir)
page := ocr.Gray(img)
line := r.ReadLine(page, ocr.Box{X0: 120, Y0: 340, X1: 980, Y1: 372})
line.Text       // the text
line.Chars      // each character with its x span on the page
line.Confidence // the lowest probability among the line's steps
```

`ReadLines` reads the lines of a page in parallel. The characters' positions on the page let a redaction hide a word read in an image.

## In numbers

Character error rate (CER), greedy decoding, against Indra's recognizer (v1.1.0) read by the same Go code on the same crops. Dot leaders ("......" in a table of contents) are cut to "..." in both texts before comparing (`ocr.CompareForm`): no reader counts them right.

| Set | Lines | indecis v1 | indecis v2 | Indra |
| --- | --- | --- | --- | --- |
| Synthetic, held out | 2,000 | 2.90% | **2.58%** | 4.98% |
| HAL born-digital PDFs, French (documents not trained on) | 3,270 | 1.01% | 0.91% | **0.60%** |
| HAL born-digital PDFs, English | 3,543 | 0.55% | 0.51% | **0.38%** |
| SROIE receipts, scanned (in capitals) | 2,000 | 7.11% | 6.50% | **3.04%** |

Indra stays ahead on real documents, and well ahead on scanned receipts, which our synthetic lines imitate poorly. The synthetic set is ours: Indra's numbers there say little.

- v1: Indra's architecture with a layer norm and 206 symbols (1.06 M parameters), 20,000 steps of 64 lines (1.28 M lines: 1 M synthetic lines, 25,000 lines of HAL PDFs), 80 minutes on the 14 cores of a Core Ultra 7 265U.
- v2: v1 trained 15,000 more steps on v1's lines and 600,000 new ones (receipt lines, typographic quotes, columns of wide spaces, dot-matrix print).

int8 changes no rate by more than 0.02 point.

Reading a line 800 pixels wide on one core (P-core), decoding included: 6.8 ms in float32, **4.6 ms in int8**. Indra's recognizer (1,698 symbols) takes 8.2 ms in float32 and 5.4 ms in int8 in Go, 5.4 ms in onnxruntime (float32, one thread).

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
