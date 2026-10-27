"""Lines and pages of ALTO ground truth (HTR-United sets: eScriptorium
exports), each page's image found next to its XML.

    tools/oracle/.venv/bin/python tools/ocrsynth/alto_lines.py SRC-DIR out-dir source-name

A TextLine's box (HPOS, VPOS, WIDTH, HEIGHT), shifted with all the page's
by the offset that puts the most ink inside them (some sets' coordinates
are off their image), is tightened to its ink, as the other converters do;
its text is its Strings joined by spaces:
out-dir/lines.jsonl (ocr.Sample), out-dir/pages.jsonl (ocr.PageSample).
Image paths are absolute.
"""

import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF")


def local(tag):
    return tag.rsplit("}", 1)[-1]


def find_image(xml_path, name):
    d = os.path.dirname(xml_path)
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    candidates = []
    if name:
        candidates += [os.path.join(d, name), os.path.join(d, "..", "images", name), os.path.join(d, "..", "img", name)]
    for ext in EXTS:
        candidates += [os.path.join(d, stem + ext), os.path.join(d, "..", "images", stem + ext), os.path.join(d, "..", "img", stem + ext)]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    if name:  # anywhere under the set
        hits = glob.glob(os.path.join(d, "..", "**", name), recursive=True)
        if hits:
            return os.path.abspath(hits[0])
    return None


def ink_box(gray, b):
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


def baseline(attr):
    """A BASELINE attribute ("x1 y1 x2 y2 ...") as (x0, x1, y), or None."""
    try:
        v = [float(t) for t in (attr or "").replace(",", " ").split()]
    except ValueError:
        return None
    if len(v) < 4 or len(v) % 2:
        return None
    xs, ys = v[0::2], v[1::2]
    return (min(xs), max(xs), sum(ys) / len(ys))


def boxes_from_baselines(raw):
    """Lines given by their baseline become boxes from 0.75 of the page's
    median line pitch above it to 0.25 below: the boxes eScriptorium writes
    hold its line masks, a line and a half high."""
    ys = sorted(b[2] for b, _ in raw if len(b) == 3)
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 5]
    if not gaps:
        return [(b, t) for b, t in raw if len(b) == 4]
    pitch = float(np.median(gaps))
    out = []
    for b, t in raw:
        if len(b) == 3:
            x0, x1, y = b
            b = [x0, y - 0.75 * pitch, x1, y + 0.25 * pitch]
        out.append((b, t))
    return out


def realign(gray, raw):
    """The (dx, dy) that puts the most ink inside the boxes: some sets'
    coordinates are offset from their image (an image cropped or rescaled
    after the transcription). Searched over half a line height, by steps of
    a tenth."""
    if not raw:
        return 0, 0
    lo, hi = np.percentile(gray, (2, 98))
    ink = (gray < (lo + hi) / 2).astype(np.int64)
    integral = ink.cumsum(0).cumsum(1)
    H, W = ink.shape

    def total(dx, dy):
        s = 0
        for x0, y0, x1, y1 in raw:
            a, b = min(max(int(x0 + dx), 0), W - 1), min(max(int(y0 + dy), 0), H - 1)
            c, d = min(max(int(x1 + dx), 0), W - 1), min(max(int(y1 + dy), 0), H - 1)
            s += integral[d, c] - integral[b, c] - integral[d, a] + integral[b, a]
        return s

    h = float(np.median([y1 - y0 for _, y0, _, y1 in raw]))
    steps = [round(k * h / 10) for k in range(-5, 6)]
    best = max(((total(0, dy), dy) for dy in steps))[1]
    bestx = max(((total(dx, best), dx) for dx in steps))[1]
    return bestx, best


def main():
    src, out, source = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    lines_f = open(os.path.join(out, "lines.jsonl"), "a", encoding="utf-8")
    pages_f = open(os.path.join(out, "pages.jsonl"), "a", encoding="utf-8")
    n_pages = n_lines = missing = 0
    for xml_path in sorted(glob.glob(os.path.join(src, "**", "*.xml"), recursive=True)):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        if local(root.tag) != "alto":
            continue
        name = None
        for el in root.iter():
            if local(el.tag) == "fileName" and el.text:
                name = os.path.basename(el.text.strip())
                break
        img = find_image(xml_path, name)
        if img is None:
            missing += 1
            continue
        try:
            gray = np.asarray(Image.open(img).convert("L"))
        except (Image.DecompressionBombError, OSError):
            missing += 1  # a page too large to decode safely, or unreadable
            continue
        raw = []
        for tl in root.iter():
            if local(tl.tag) != "TextLine":
                continue
            words = [s.get("CONTENT", "") for s in tl if local(s.tag) == "String"]
            text = " ".join(" ".join(words).split())
            base = baseline(tl.get("BASELINE"))
            if base is not None and text:
                raw.append((base, text))
                continue
            try:
                x, y, w, h = (float(tl.get(k)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
            except (TypeError, ValueError):
                continue
            if text and w > 0 and h > 0:
                raw.append(([x, y, x + w, y + h], text))
        raw = boxes_from_baselines(raw)
        dx, dy = realign(gray, [b for b, _ in raw])
        boxes, texts = [], []
        for (x0, y0, x1, y1), text in raw:
            b = ink_box(gray, [x0 + dx, y0 + dy, x1 + dx, y1 + dy])
            if b is None or b[3] - b[1] < 4:
                continue
            boxes.append(b)
            texts.append(text)
            lines_f.write(json.dumps({"image": img, "box": b, "text": text, "meta": {"source": source}}, ensure_ascii=False) + "\n")
            n_lines += 1
        if boxes:
            pages_f.write(json.dumps({"image": img, "boxes": boxes, "texts": texts, "meta": {"source": source}}, ensure_ascii=False) + "\n")
            n_pages += 1
    print(f"{source}: {n_pages} pages, {n_lines} lines, {missing} pages without a readable image")


if __name__ == "__main__":
    main()
