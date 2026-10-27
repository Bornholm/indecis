# OCR data: sources and licenses

Data used to train and measure the line recognizer (see [ocr-plan.md](ocr-plan.md)). None of it is versioned in this repository; the commands below rebuild it under `~/.cache/indecis/datasets/ocr`.

## Training

| Source | License | Use |
| --- | --- | --- |
| Synthetic lines (`tools/ocrsynth`) | ours | Main training source |
| Fonts: Google Fonts (41 families, `ofl/`) | SIL Open Font License 1.1 | Rendering |
| Fonts: DejaVu, Liberation, Noto (Sans, Serif, Mono, Display), URW base35 | Bitstream Vera / OFL / OFL / AGPL with font exception | Rendering |
| French and English Wikipedia (20231101, first parquet shard of each) | CC BY-SA 4.0 | Running text rendered into images, never shipped |
| ASN letters, Enron (French translations and originals), from the email proof of concept | public documents / public corpus | Administrative and business text, rendered |
| HAL born-digital PDFs, documents not kept for evaluation (`tools/ocrsynth/ocrpdf`): 234, then 412 French and 494 English more documents | CC BY 4.0, CC0, Licence Ouverte 2.0 | Real layouts and fonts, text layer as ground truth: 129,000 lines, 2,505 pages |
| DocLayNet v1.1, first train shard (`docling-project/DocLayNet-v1.1`) | CDLA-Permissive-1.0 | Reports, manuals, patents, laws, tenders: rendered pages with their PDF text cells (`tools/ocrsynth/doclaynet_lines.py`) |
| Synthetic pages (`ocrsynth -pages`): columns, headings, paragraphs, lists, tables | ours | Line detection: 3,000 pages |
| CORD v2, train split (`naver-clova-ix/cord-v2`, 800 receipts) | CC BY 4.0 | Photographed receipts: 12,623 lines, 800 pages (`tools/ocrsynth/cord_lines.py`); their texts also feed the language model |

Fonts only shape the rendered images; the model contains none of them. Variable fonts are instantiated at weights 400 and 700 with `fonttools varLib.instancer` (x/image cannot read variable fonts' metrics).

## Evaluation

| Set | Lines | License | Notes |
| --- | --- | --- | --- |
| Synthetic, seed 900001 | 2,000 | ours | Same distribution as training: checkpoint selection |
| HAL born-digital, French | 3,270 lines, 80 pages (38 documents) | CC BY 4.0, CC0, Licence Ouverte 2.0 | Documents never used in training; pages rendered at 100 to 300 dpi |
| HAL born-digital, English | 3,543 lines, 73 pages (40 documents) | same | same |
| CORD v2, test split | 1,503 lines, 100 pages | CC BY 4.0 | Held out to set the language model's weights, never reported |
| HAL born-digital, 99 English documents of the extra sample | 3,000 lines drawn from 18,161 | CC BY 4.0, CC0, Licence Ouverte 2.0 | Held out to set the language model's weights, never reported |
| CORD v2, validation split | 1,413 lines, 100 pages | CC BY 4.0 | Photographed receipts (Indonesian), a row's words cut at gaps wider than 1.5 word heights |
| SROIE 2019, test (`priyank-m/SROIE_2019_text_recognition`) | 2,000 drawn from 18,704 | research only (ICDAR 2019 competition) | Scanned receipts, English; references in capitals: measured with `-fold-case`; local use only |

HAL adds a cover page to every document: `ocrpdf` starts at page 2. Documents whose text layer maps ligatures to wrong characters ("obten;on", "CeHe") are dropped.

## Rebuilding

```bash
D=~/.cache/indecis/datasets/ocr
# Text: Wikipedia shards from huggingface.co/datasets/wikimedia/wikipedia
tools/oracle/.venv/bin/python tools/ocrsynth/wiki_text.py fr0.parquet $D/text/wiki_fr.txt 400000
tools/oracle/.venv/bin/python tools/ocrsynth/wiki_text.py en0.parquet $D/text/wiki_en.txt 300000

# Synthetic lines
cd tools/ocrsynth
go run . -fonts ~/.cache/indecis/fonts \
  -text $D/text/wiki_fr.txt,$D/text/wiki_fr.txt,$D/text/wiki_en.txt,$D/text/admin_fr.txt,$D/text/enron_en.txt \
  -n 1000000 -shard-size 5000 -out $D/synth-v1

# Born-digital PDFs
go run ./ocrpdf -pdfs $D/hal/fr -out $D/hal-fr -lang fr
```
