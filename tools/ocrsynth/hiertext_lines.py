"""Lines and pages of HierText (github.com/google-research-datasets/hiertext,
CC BY-SA 4.0): photographs and scans of Open Images, every piece of text
outlined as words, lines and paragraphs.

    python3 tools/ocrsynth/hiertext_lines.py hiertext/validation.jsonl.gz hiertext/validation out-dir

Writes out-dir/lines.jsonl (ocr.Sample): the legible printed lines lying
within 3° of level, with the box around their outline; and
out-dir/pages.jsonl (ocr.PageSample): the images whose lines all lie
within 5° of level, none vertical, with the boxes of all their lines,
legible or not (text all the same, which the detector must find). Image
paths point into the image directory.
"""

import gzip
import json
import math
import os
import sys


def angle(vertices):
    """The slant of a line's outline, in degrees: the direction of its
    longest side among the first ones (outlines run along the top first)."""
    best, deg = 0.0, 0.0
    n = len(vertices)
    for i in range(n):
        (x0, y0), (x1, y1) = vertices[i], vertices[(i + 1) % n]
        d = math.hypot(x1 - x0, y1 - y0)
        if d > best:
            best = d
            deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # A side runs either way: fold to (-90, 90].
    while deg > 90:
        deg -= 180
    while deg <= -90:
        deg += 180
    return deg


def box(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return [min(xs), min(ys), max(xs), max(ys)]


def main():
    src, img_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    img_rel = os.path.relpath(img_dir, out)
    ann = json.load(gzip.open(src))["annotations"]
    n_lines = n_pages = 0
    with open(os.path.join(out, "lines.jsonl"), "w", encoding="utf-8") as lf, \
            open(os.path.join(out, "pages.jsonl"), "w", encoding="utf-8") as pf:
        for a in ann:
            image = os.path.join(img_rel, a["image_id"] + ".jpg")
            if not os.path.exists(os.path.join(out, image)):
                continue
            boxes, texts, level = [], [], True
            for p in a["paragraphs"]:
                for l in p["lines"]:
                    v = l["vertices"]
                    if len(v) < 3:
                        continue
                    b = box(v)
                    h = b[3] - b[1]
                    deg = angle(v)
                    if l.get("vertical") or abs(deg) > 5:
                        level = False
                        continue
                    boxes.append(b)
                    texts.append(l["text"] if l["legible"] else "")
                    if (l["legible"] and not l["handwritten"] and abs(deg) <= 3 and h >= 8
                            and l["text"].strip() and b[2] - b[0] >= h):
                        lf.write(json.dumps({"image": image, "box": b, "text": l["text"],
                                             "meta": {"source": "hiertext"}}, ensure_ascii=False) + "\n")
                        n_lines += 1
            if level and boxes:
                pf.write(json.dumps({"image": image, "boxes": boxes, "texts": texts,
                                     "meta": {"source": "hiertext"}}, ensure_ascii=False) + "\n")
                n_pages += 1
    print(f"{n_lines} lines, {n_pages} pages", file=sys.stderr)


if __name__ == "__main__":
    main()
