# OCR plan: a text-line recognizer trained in Go

Goal: read printed text from images, in French and English first, with a small model that indecis trains and runs on CPU, in pure Go. The pages read this way feed the existing text models (redaction, triage, decisions).

The starting point is [Indra OCR](https://github.com/datalab-to/indra) (Datalab): a chain of small ONNX networks run by onnxruntime from Python. Its code is Apache-2.0, but its weights are under a modified OpenRAIL-M license (no use competing with Datalab's API, revenue threshold). indecis therefore reuses its **architecture and measurements**, never its weights: they are read locally, as a reference, and are never versioned, converted for distribution or published.

## The reference recognizer (Indra `rec.onnx`, v1.1.0)

Read from the ONNX graph:

- **Input**: an ink map (0 = paper, 1 = ink) of 96 × W pixels: the line resized to 32 pixels high (`crop_line`: box padded by 0.2 × height above and below, 0.3 × height left and right), in a window of three line heights. W is padded to a multiple of 8; the true width gives the number of valid steps, ⌈w/4⌉.
- **Stem**: the middle band (rows 32 to 64) cut into 32 × 4 patches, projected to 128 channels: one step every 4 pixels.
- **Context branch**: 4 × 4 patches of the whole window (32 channels, HardSwish), a 3 × 3 depthwise convolution, a pointwise convolution to 64 (HardSwish), then attention over the 24 rows (4 heads of 16 channels, learned position bias). The pooled summary is projected to 128 and added to the stem.
- **Body**: BatchNorm, HardSwish, mask, then 10 blocks: depthwise 1D convolution (k = 9), expansion 128 → 256, HardSwish, squeeze-excitation over the valid steps (256 → 64 → 256), projection 256 → 128, residual, mask.
- **Head**: 128 → 1,698 symbols (blank = 0), CTC.
- 1.25 M parameters, about 1.9 MFLOP per step: 0.4 GFLOP for a line of 800 pixels.

Measured on the Core Ultra 7 265U: onnxruntime, 1 thread, 2.0 ms for a line of 256 pixels, 5.7 ms for 800, 10.9 ms for 1,600. indecis's float32 GEMM: 80 to 95 GFLOPS on one P-core, 310 to 345 GFLOPS on 14 cores.

## Status (October 27, 2026)

P1 to P8 done, P10 measured (see [ocr.md](ocr.md)). Recognizer v5 with its language model reads ahead of Indra on French and English PDFs, receipts photographed and old French prints, behind on scanned English receipts. Page reading, end to end, is ahead of Indra on English PDFs, receipts and old prints, behind on French PDFs, where the detector misses table cells. On Omni Parse Bench's Latin printed image pages, 51.4 against Indra's 63.5. Next: tables and forms (cells kept apart), more real scans, longer training.

## Phases

Each phase ends with a commit (or several) on `feat/ocr` and a measured acceptance criterion.

### P1. Reference and oracle

- `tools/oracle/ocr_rec.py`: the architecture above in PyTorch, driven by a config (widths, number of blocks, vocabulary, normalization `batch` or `layer`).
- `tools/oracle/ocr_indra_check.py` (local only): loads `rec.onnx`'s weights into the PyTorch module and compares it with onnxruntime on real and random lines. It also writes these weights as safetensors under `~/.cache/indecis/models/indra-rec` for the Go tests, outside the repository.
- `tools/oracle/ocr_fixtures.py`: a tiny model with random weights (2 blocks, small vocabulary), its inputs, logits and CTC-loss gradients, versioned in `testdata/ocr-tiny`.

Acceptance: PyTorch and onnxruntime agree within 1e-4 on Indra's weights.

### P2. Forward in Go (`internal/linerec`)

- Config and safetensors loading; BatchNorm folded into a scale and a shift.
- Operators: patch projection (reshape + GEMM), depthwise convolutions 1D and 2D (portable SIMD), HardSwish, masked squeeze-excitation, height attention, head, greedy CTC decoding with confidences.
- Lines of a batch at their own width, no padding beyond the multiple of 4.

Acceptance: parity with the fixtures (1e-5) and, locally, with onnxruntime on Indra's weights; single-thread latency within 30% of onnxruntime.

### P3. Public package `ocr` and command

- Line preprocessing in Go: crop with padding, light-on-dark detection, area and bilinear resizing, ink map.
- `ocr.Recognizer`: `ReadLine(img, box)`, `ReadLines`, parallel over lines.
- Metrics: character and word error rates (CER, WER), on an evaluation file (JSONL: image, box, text).
- `indecis ocr read` and `indecis ocr eval`.

Acceptance: on sample lines, the Go path with Indra's weights reads what Indra's Python reads.

### P4. Training in Go

- Backward of every operator, checked by finite differences, like `modernbert`.
- CTC loss in log space (forward-backward), checked against `torch.nn.functional.ctc_loss`.
- Training-step parity with PyTorch on the tiny model (gradients, clipping, AdamW).
- Normalization `layer` (LayerNorm over channels, per step): BatchNorm with masked widths depends on the batch; the trained model keeps the same inference code.
- Loop: batches grouped by width, warmup then cosine decay, clipping, checkpoints, reductions in fixed order (same model whatever the number of cores).
- `indecis ocr train`.

Acceptance: step parity with PyTorch; a small set learned to a CER of 0.

### P5. Synthetic lines (`tools/ocrsynth`, separate module)

A separate module because rendering fonts needs `golang.org/x/image`, and the library depends on nothing outside the standard library.

- Fonts under free licenses (OFL, Apache, Bitstream Vera/DejaVu, Liberation): serif, sans, monospace, condensed, bold, italic.
- Texts: French and English sentences (Wikipedia, CC BY-SA), and generated strings: numbers, amounts, dates, references, addresses, emails, URLs, uppercase, punctuation, French typography (« », non-breaking spaces, œ, apostrophes).
- Degradations: blur, noise, contrast, slight rotation and shear, erosion and dilation, JPEG, scan-like background, neighbouring lines in the window, box jitter.
- Output: shards of ink maps already at 96 pixels high (uint8), with their text.

Acceptance: at least 1,000 lines per second on all cores; a visual review of a sample.

### P6. Real evaluation sets

- French printed lines with ground truth under a free license (HTR-United catalogue, CC BY datasets).
- Born-digital PDFs under open licenses (data.gouv.fr Licence Ouverte, HAL and arXiv CC BY): page rendered with poppler, text layer as ground truth, lines cut with `crop_line`'s rule. Also a training source.
- A few real scans (printed then scanned pages) for the degradations synthesis misses.
- Each source is documented with its license in `docs/ocr-data.md`.

Acceptance: at least 2,000 French and 2,000 English test lines, sources and licenses listed.

### P7. Training and comparison

- First model on synthetic lines, then synthetic + born-digital PDFs.
- CER and WER against Indra (local reference) on the P6 sets, per source; ablations: data mix, vocabulary, number of blocks, training length.
- int8 inference (VNNI kernel), and its effect on the CER.

Acceptance: a measured report, with Indra's numbers on the same sets.

### P8. Line detection and pages

Indra's `det.onnx` (91 k parameters), read from its graph: the page in gray, long side 1,024 pixels; a 3 × 3 stride-2 stem (16 channels, HardSwish); four stages of MobileNet blocks (depthwise 3 × 3 then 5 × 5, expansion, projection, residual) at strides 4, 8, 16 and 32 (24, 32, 48, 64 channels); an FPN (1 × 1 laterals to 24 channels, nearest upsampling, sums); a head of three maps (text, picture, and a third with a Softplus), brought back to full resolution by DepthToSpace.

- Operators to add to Go, forward and backward: dense 3 × 3 convolution with stride (im2col + GEMM), depthwise 2D 3 × 3 and 5 × 5 with stride 1 or 2 (generalizing the context branch's), nearest upsampling, DepthToSpace, Softplus.
- Targets, DBNet style: the line boxes shrunk to their core, a picture mask.
- Training pages: born-digital PDFs (line boxes from the text layer, `ocrpdf` already cuts them), synthetic pages composed of rendered paragraphs with the line generator's damage, DocLayNet (CDLA-Permissive-1.0) for figures and tables. HierText (CC BY-SA 4.0) only after a decision on share-alike.
- Post-processing: threshold, connected components, boxes grown back, deskew; reading order by XY-cut first.
- `ocr.ReadPage`, `indecis ocr page`.
- Measures: line detection F1 at IoU 0.5 on HAL pages, and end-to-end CER of a page (its text in reading order against the text layer).

### P10. Indra's benchmarks

Measure indecis where Indra publishes its numbers, so that the two can be compared on its own ground.

- [OmniParseBench](https://github.com/datalab-to/omni_parse_bench) (Apache-2.0 code; tests CC BY 4.0, ODC-BY and MIT by source; data on Hugging Face `datalab-to/omni_parse_bench`): 2,937 pages, pass/fail tests in three families. indecis outputs lines, so only the **text** family applies (`present`, `order`, `repeat`), as for Tesseract in Indra's table; tables and layout stay unscored. Compare with Indra's OCR-only text score (40.7: no text layer, as indecis), on all pages and on the Latin-script ones (Indra: 61.6).
- olmOCR-bench (ODC-BY), where Indra reports headers and footers, tables, multi-column pages and old scans: the text-presence, absence and reading-order tests apply.
- Needs first: reading order (XY-cut, columns), lines grouped into paragraphs, pages rendered from the PDFs (pdftoppm), and the page output as HTML for the benchmark's scorer (`opb score --pred outputs.json --gold`), run as published so the numbers compare.

### P9. Documentation

`docs/ocr.md` (usage, numbers), `docs/architecture.md`, `ROADMAP.md`, README.

## Open questions

- Distribution of a trained model: same releases as the other models, or a separate repository?
- HierText and other share-alike datasets for training: accepted or not?
- Browser target (wasm, redaction demo): to keep in mind when choosing the preprocessing, no specific work before P7.
