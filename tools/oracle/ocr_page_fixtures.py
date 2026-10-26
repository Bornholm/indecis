"""Page fixtures of the OCR: a rendered page, its line boxes and texts, and
each line cut by Indra OCR's crop_line (OpenCV), for the Go crop: crops.u8,
the ink maps in 255ths, line after line.

    tools/oracle/.venv/bin/python tools/oracle/ocr_page_fixtures.py --out testdata/ocr-page

crop_line below is Indra OCR's (indra_ocr/rec_preproc.py, Apache-2.0),
without its options: three-line window, margins 0.2 and 0.3.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LINES = [
    # text, font, size, dark
    ("Le colis n'est pas arrivé, je veux parler à quelqu'un.", "/usr/share/fonts/TTF/DejaVuSerif.ttf", 40, False),
    ("Facture n° 2026-0412 : 1 234,56 € TTC", "/usr/share/fonts/TTF/DejaVuSans.ttf", 30, False),
    ("ÉTAT DES LIEUX", "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 56, True),
    ("The quick brown fox jumps over the lazy dog.", "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 22, False),
    ("ref: AZ-778/b  tel. 01 23 45 67 89", "/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 18, False),
]

HEIGHT, WINDOW, PAD_Y, PAD_X = 32, 3, 0.2, 0.3


def is_dark_line(gray, box) -> bool:
    H, W = gray.shape
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    b = gray[max(0, int(y0)) : min(H, int(y1) + 1), max(0, int(x0)) : min(W, int(x1) + 1)]
    lo, med, hi = np.percentile(b, (2, 50, 98))
    return bool(hi - med > 2 * (med - lo) + 8)


def crop_line(gray, box, max_width=4096):
    pad_y = (WINDOW * (1 + 2 * PAD_Y) - 1) / 2
    height = HEIGHT * WINDOW
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    h = max(1.0, y1 - y0)
    X0, X1 = x0 - PAD_X * h, x1 + PAD_X * h
    Y0, Y1 = y0 - pad_y * h, y1 + pad_y * h
    H, W = gray.shape
    ix0, iy0 = int(math.floor(X0)), int(math.floor(Y0))
    ix1, iy1 = int(math.ceil(X1)), int(math.ceil(Y1))
    cw, ch = max(1, ix1 - ix0), max(1, iy1 - iy0)
    sx0, sy0, sx1, sy1 = max(0, ix0), max(0, iy0), min(W, ix1), min(H, iy1)
    dark = is_dark_line(gray, box)
    crop = np.full((ch, cw), 0 if dark else 255, np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = gray[sy0:sy1, sx0:sx1]
    out_w = max(8, min(max_width, round(cw * height / ch)))
    interp = cv2.INTER_AREA if ch > height else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (out_w, height), interpolation=interp)
    ink = resized.astype(np.float32) / 255.0
    return (ink if dark else 1.0 - ink), dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    W, margin, gap = 1400, 60, 40
    img = Image.new("L", (W, 900), 255)
    d = ImageDraw.Draw(img)
    y = margin
    cases = []
    for text, path, size, dark in LINES:
        font = ImageFont.truetype(path, size)
        x0, y0, x1, y1 = font.getbbox(text)
        if dark:
            d.rectangle((margin - 30, y - 30, margin + x1 - x0 + 30, y + y1 - y0 + 30), fill=20)
        d.text((margin - x0, y - y0), text, font=font, fill=235 if dark else 0)
        box = [margin, y, margin + x1 - x0, y + y1 - y0]
        cases.append({"text": text, "box": box})
        y += (y1 - y0) + gap + (60 if dark else 0)
    img = img.crop((0, 0, W, y + margin))
    img.save(a.out / "page.png")

    gray = np.asarray(img)
    inks = []
    for c in cases:
        ink, dark = crop_line(gray, c["box"])
        c["dark"], c["width"] = dark, ink.shape[1]
        inks.append(ink.ravel())
    # Ink in 255ths: OpenCV resizes in uint8, the values are exact.
    u8 = np.rint(np.concatenate(inks) * 255).astype(np.uint8)
    (a.out / "crops.u8").write_bytes(u8.tobytes())
    (a.out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(json.dumps([(c["width"], c["dark"]) for c in cases]))


if __name__ == "__main__":
    main()
