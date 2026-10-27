"""Regions of labeled pages whose ink no line covers, written as the pages'
ignore regions (ocr.PageSample.Ignore): pages taken from a PDF's text layer
label the text of their pictures and vector charts nowhere, and the
detector, shown it as paper, would learn to skip text in figures.

    tools/oracle/.venv/bin/python tools/ocrsynth/ignore_unlabeled.py pages.jsonl [more.jsonl ...]

Each file is rewritten in place (the original kept as .orig). The ink
outside the line boxes (grown by a third of their height) is grouped by
proximity, rules aside; a group of some size becomes a region. Regions already given
(DocLayNet's pictures) are kept.
"""

import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def regions(gray, boxes):
    h, w = gray.shape
    lo, paper = np.percentile(gray, (1, 50))
    if paper - lo < 40:
        return []
    ink = gray < (lo + paper) / 2
    covered = np.zeros_like(ink)
    for x0, y0, x1, y1 in boxes:
        m = (y1 - y0) / 3
        covered[max(0, int(y0 - m)):min(h, int(y1 + m) + 1), max(0, int(x0 - m)):min(w, int(x1 + m) + 1)] = True
    rest = ink & ~covered
    # Rules (of tables, under headings) are no text: the detector must keep
    # learning to pass them by. Long thin strokes are taken out first.
    k = max(20, int(0.03 * max(h, w)))
    rules = ndimage.binary_opening(rest, structure=np.ones((1, k))) | ndimage.binary_opening(rest, structure=np.ones((k, 1)))
    rest &= ~ndimage.binary_dilation(rules, iterations=2)
    if rest.sum() < 30:
        return []
    # Parts of one figure lie close together: grow them before grouping.
    grown = ndimage.binary_dilation(rest, iterations=max(4, int(0.01 * max(h, w))))
    labels, n = ndimage.label(grown)
    out = []
    for sl in ndimage.find_objects(labels):
        ys, xs = sl
        bw, bh = xs.stop - xs.start, ys.stop - ys.start
        if bw * bh < 0.002 * h * w or rest[sl].sum() < 30:
            continue
        out.append([int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)])
    return out


def main():
    for path in sys.argv[1:]:
        base = os.path.dirname(path)
        orig = path + ".orig"
        if not os.path.exists(orig):
            shutil.copy(path, orig)
        pages = [json.loads(l) for l in open(orig, encoding="utf-8")]
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for p in pages:
                im = Image.open(os.path.join(base, p["image"])).convert("L")
                # Found on the page reduced to about 1,000 pixels, then
                # scaled back.
                k = max(1.0, max(im.size) / 1000)
                if k > 1:
                    im = im.resize((round(im.width / k), round(im.height / k)), Image.BOX)
                g = np.asarray(im, dtype=np.float32)
                found = [[round(v * k) for v in r] for r in regions(g, [[v / k for v in b] for b in p["boxes"]])]
                if found:
                    p["ignore"] = p.get("ignore", []) + found
                    n += 1
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{path}: {n} of {len(pages)} pages with regions to ignore", file=sys.stderr)


if __name__ == "__main__":
    main()
