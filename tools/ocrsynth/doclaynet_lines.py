"""Lines and pages of DocLayNet v1.1 (huggingface.co/datasets/docling-project/
DocLayNet-v1.1, CDLA-Permissive-1.0): 1,025-pixel renders of PDF pages
(reports, manuals, patents, laws, tenders, articles) with the text cells of
their PDF, about one line each.

    tools/oracle/.venv/bin/python tools/ocrsynth/doclaynet_lines.py train-00000.parquet out-dir [max-pages]

A cell's box (COCO [x, y, w, h]) is tightened to the dark pixels inside it,
as ocrpdf does; cells without ink are dropped: out-dir/lines.jsonl
(ocr.Sample), out-dir/pages.jsonl (ocr.PageSample), out-dir/pages/*.png.
"""

import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def ink_box(gray, b):
    x, y, w, h = b
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(gray.shape[1], int(np.ceil(x + w))), min(gray.shape[0], int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.nonzero(gray[y0:y1, x0:x1] < 128)
    if len(xs) == 0:
        return None
    return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]


def main():
    src, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 30
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    lines_f = open(os.path.join(out, "lines.jsonl"), "a", encoding="utf-8")
    pages_f = open(os.path.join(out, "pages.jsonl"), "a", encoding="utf-8")
    shard = os.path.basename(src).split(".")[0]
    n_pages = n_lines = 0
    for batch in pq.ParquetFile(src).iter_batches(batch_size=32, columns=["image", "pdf_cells", "metadata"]):
        for row in batch.to_pylist():
            if n_pages >= limit:
                break
            meta = row["metadata"]
            gray = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
            name = f"dln-{shard}-{meta['image_id']:06d}.png"
            gray.save(os.path.join(out, "pages", name))
            g = np.asarray(gray)
            boxes, texts = [], []
            for element in row["pdf_cells"] or []:
                for cell in element or []:
                    text = " ".join((cell.get("text") or "").split())
                    if not text:
                        continue
                    b = ink_box(g, cell["bbox"])
                    if b is None or b[3] - b[1] < 3:
                        continue
                    boxes.append(b)
                    texts.append(text)
                    m = {"source": "doclaynet", "category": meta["doc_category"]}
                    lines_f.write(json.dumps({"image": "pages/" + name, "box": b, "text": text, "meta": m}, ensure_ascii=False) + "\n")
                    n_lines += 1
            pages_f.write(json.dumps({"image": "pages/" + name, "boxes": boxes, "texts": texts,
                                      "meta": {"source": "doclaynet", "category": meta["doc_category"]}}, ensure_ascii=False) + "\n")
            n_pages += 1
    print(n_pages, "pages,", n_lines, "lines")


if __name__ == "__main__":
    main()
