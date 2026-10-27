"""Lines and pages of CORD v2 (huggingface.co/datasets/naver-clova-ix/cord-v2,
CC BY 4.0): scanned and photographed receipts, words with their boxes.

    tools/oracle/.venv/bin/python tools/ocrsynth/cord_lines.py cord/validation-*.parquet out-dir

The words of a receipt row (row_id) form a line, cut where two words are
more than 1.5 word heights apart (an item and its price, at the ends of the
row): out-dir/lines.jsonl (ocr.Sample), out-dir/pages.jsonl
(ocr.PageSample), out-dir/pages/*.png.
"""

import io
import json
import os
import sys
from collections import defaultdict

import pyarrow.parquet as pq
from PIL import Image


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    t = pq.read_table(src)
    lines_f = open(os.path.join(out, "lines.jsonl"), "a", encoding="utf-8")
    pages_f = open(os.path.join(out, "pages.jsonl"), "a", encoding="utf-8")
    split = os.path.basename(src).split("-")[0]
    n_lines = 0
    for i in range(t.num_rows):
        img = t.column("image")[i].as_py()
        gt = json.loads(t.column("ground_truth")[i].as_py())
        name = f"cord-{split}-{i:04d}.png"
        im = Image.open(io.BytesIO(img["bytes"])).convert("L")
        im.save(os.path.join(out, "pages", name))
        rows = defaultdict(list)
        for vl in gt["valid_line"]:
            for w in vl["words"]:
                q = w["quad"]
                xs, ys = [q["x1"], q["x2"], q["x3"], q["x4"]], [q["y1"], q["y2"], q["y3"], q["y4"]]
                if w["text"].strip():
                    rows[w["row_id"]].append((min(xs), min(ys), max(xs), max(ys), w["text"].strip()))
        boxes, texts = [], []
        for words in rows.values():
            words.sort()
            h = sorted(w[3] - w[1] for w in words)[len(words) // 2]
            seg = [words[0]]
            for w in words[1:] + [None]:
                if w is None or w[0] - seg[-1][2] > 1.5 * h:
                    box = [min(s[0] for s in seg), min(s[1] for s in seg), max(s[2] for s in seg), max(s[3] for s in seg)]
                    text = " ".join(s[4] for s in seg)
                    boxes.append(box)
                    texts.append(text)
                    lines_f.write(json.dumps({"image": "pages/" + name, "box": box, "text": text,
                                              "meta": {"source": "cord", "split": split}}, ensure_ascii=False) + "\n")
                    n_lines += 1
                    seg = [w] if w else []
                else:
                    seg.append(w)
        pages_f.write(json.dumps({"image": "pages/" + name, "boxes": boxes, "texts": texts,
                                  "meta": {"source": "cord", "split": split}}, ensure_ascii=False) + "\n")
    print(t.num_rows, "receipts,", n_lines, "lines")


if __name__ == "__main__":
    main()
