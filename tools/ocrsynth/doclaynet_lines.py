"""Lines and pages of DocLayNet v1.1 (huggingface.co/datasets/docling-project/
DocLayNet-v1.1, CDLA-Permissive-1.0): 1,025-pixel renders of PDF pages
(reports, manuals, patents, laws, tenders, articles) with the text cells of
their PDF, about one line each.

    tools/oracle/.venv/bin/python tools/ocrsynth/doclaynet_lines.py train-00000.parquet out-dir [max-pages]

The cells of a line (the PDF cuts lines into runs, sometimes mid-word) are
merged, and each line's box tightened to its ink, as ocrpdf does; lines
without ink are dropped, and pages whose lines miss text are too: out-dir/lines.jsonl
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
    """b ([x0, y0, x1, y1]) tightened to its ink: pixels darker than the
    middle of the box's 2nd and 98th percentiles (grey captions too)."""
    x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
    x1, y1 = min(gray.shape[1], int(np.ceil(b[2]))), min(gray.shape[0], int(np.ceil(b[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = gray[y0:y1, x0:x1]
    lo, hi = np.percentile(patch, (2, 98))
    if hi - lo < 40:
        return None
    ys, xs = np.nonzero(patch < (lo + hi) / 2)
    if len(xs) == 0:
        return None
    return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]


PICTURE, TABLE = 7, 9  # DocLayNet's COCO categories


def coverage(gray, elements, boxes):
    """The lowest share, over the page's text blocks (pictures aside), of
    the block's ink inside the line boxes, relative to what the block's kind
    allows (0.9, tables 0.75: bits of their rules stay): the PDF's cells
    sometimes leave words out, and a page with unboxed text would teach the
    detector to skip text. Below 1, the page misses text."""
    covered = np.zeros(gray.shape, bool)
    for x0, y0, x1, y1 in boxes:
        covered[y0:y1, x0:x1] = True
    worst = 1.0
    for (x, y, w, h), cat in elements:
        if cat == PICTURE:
            continue
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(gray.shape[1], int(np.ceil(x + w))), min(gray.shape[0], int(np.ceil(y + h)))
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        lo, hi = np.percentile(patch, (2, 98))
        if hi - lo < 40:
            continue
        ink = patch < (lo + hi) / 2
        # Table rules are ink no line covers: drop the rows and columns
        # mostly inked, and the pixels next to them.
        rules_r = np.convolve(ink.mean(axis=1) > 0.5, np.ones(3), "same") > 0
        rules_c = np.convolve(ink.mean(axis=0) > 0.5, np.ones(3), "same") > 0
        ink[rules_r, :] = False
        ink[:, rules_c] = False
        n = ink.sum()
        if n > 50:
            allowed = 0.75 if cat == TABLE else 0.9
            worst = min(worst, (ink & covered[y0:y1, x0:x1]).sum() / n / allowed)
    return worst


def merge_cells(cells):
    """Join the cells of one line: the PDF cuts lines into runs, sometimes
    mid-word. Cells overlapping vertically over most of the smaller height
    and less than a height apart are merged; a space goes between them
    when the gap is wider than 0.15 height."""
    lines = []  # [x0, y0, x1, y1, [(x0, x1, text)]]
    for x0, y0, x1, y1, t in sorted(cells, key=lambda c: (c[1] + c[3]) / 2):
        for ln in lines:
            h = min(y1 - y0, ln[3] - ln[1])
            over = min(y1, ln[3]) - max(y0, ln[1])
            gap = max(x0, ln[0]) - min(x1, ln[2])
            if over > 0.6 * h and gap < 1.0 * h:
                ln[0], ln[1], ln[2], ln[3] = min(ln[0], x0), min(ln[1], y0), max(ln[2], x1), max(ln[3], y1)
                ln[4].append((x0, x1, t))
                break
        else:
            lines.append([x0, y0, x1, y1, [(x0, x1, t)]])
    out = []
    for x0, y0, x1, y1, parts in lines:
        parts.sort()
        h = y1 - y0
        text = parts[0][2]
        for (a0, a1, _), (b0, b1, t) in zip(parts, parts[1:]):
            text += (" " if b0 - a1 > 0.15 * h else "") + t
        out.append(([x0, y0, x1, y1], " ".join(text.split())))
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 30
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    lines_f = open(os.path.join(out, "lines.jsonl"), "a", encoding="utf-8")
    pages_f = open(os.path.join(out, "pages.jsonl"), "a", encoding="utf-8")
    shard = os.path.basename(src).split(".")[0]
    n_pages = n_lines = dropped = 0
    for batch in pq.ParquetFile(src).iter_batches(batch_size=32, columns=["image", "pdf_cells", "metadata", "bboxes", "category_id"]):
        for row in batch.to_pylist():
            if n_pages >= limit:
                break
            meta = row["metadata"]
            gray = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
            name = f"dln-{shard}-{meta['image_id']:06d}.png"
            gray.save(os.path.join(out, "pages", name))
            g = np.asarray(gray)
            boxes, texts = [], []
            cells = []
            for element in row["pdf_cells"] or []:
                for cell in element or []:
                    text = " ".join((cell.get("text") or "").split())
                    x, y, w, h = cell["bbox"]
                    if text and w > 0 and h > 0:
                        cells.append((x, y, x + w, y + h, text))
            for box, text in merge_cells(cells):
                b = ink_box(g, box)
                if b is None or b[3] - b[1] < 3:
                    continue
                boxes.append(b)
                texts.append(text)
            if not boxes or coverage(g, zip(row["bboxes"], row["category_id"]), boxes) < 1:
                os.remove(os.path.join(out, "pages", name))
                dropped += 1
                continue
            m = {"source": "doclaynet", "category": meta["doc_category"]}
            for b, text in zip(boxes, texts):
                lines_f.write(json.dumps({"image": "pages/" + name, "box": b, "text": text, "meta": m}, ensure_ascii=False) + "\n")
                n_lines += 1
            # Text drawn inside pictures has no cells: the detector must not
            # learn it is paper.
            ignore = [[x, y, x + w, y + h] for (x, y, w, h), cat in zip(row["bboxes"], row["category_id"]) if cat == PICTURE]
            page = {"image": "pages/" + name, "boxes": boxes, "texts": texts,
                    "meta": {"source": "doclaynet", "category": meta["doc_category"]}}
            if ignore:
                page["ignore"] = ignore
            pages_f.write(json.dumps(page, ensure_ascii=False) + "\n")
            n_pages += 1
    print(n_pages, "pages,", n_lines, "lines,", dropped, "pages dropped (text the cells miss)")


if __name__ == "__main__":
    main()
