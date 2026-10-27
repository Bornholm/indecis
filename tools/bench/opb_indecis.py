"""Predictions of indecis on Omni Parse Bench (github.com/datalab-to/
omni_parse_bench), for its own scorer.

    tools/oracle/.venv/bin/python tools/bench/opb_indecis.py \
        --dataset ~/.cache/indecis/datasets/ocr/opb --out runs/opb \
        --det DET --model REC [--lm LM] [--dpi 200]
    opb grade runs/opb --dataset ~/.cache/indecis/datasets/ocr/opb [--partial]

Each page (a PDF page rendered by pdftoppm, or an image) is read by
`indecis ocr page -format html`: its lines in reading order, grouped into
paragraphs. indecis writes no tables nor layout blocks, so only the text
family (present, order, repeat) scores, as for the line-only engines of
Indra OCR's table. Pages already predicted are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--indecis", default="bin/indecis")
    ap.add_argument("--det", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lm", default="")
    ap.add_argument("--extra", default="", help="more flags for indecis ocr page, space-separated")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()

    ds = os.path.expanduser(a.dataset)
    pred_dir = os.path.join(a.out, "predictions")
    img_dir = os.path.join(a.out, "images")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    rows = pq.read_table(os.path.join(ds, "manifest.parquet")).to_pylist()

    todo = []
    for r in rows:
        src = os.path.join(ds, r["doc_path"])
        if not os.path.exists(src) or os.path.exists(os.path.join(pred_dir, r["page_id"] + ".json")):
            continue
        if src.endswith(".pdf"):
            png = os.path.join(img_dir, r["page_id"] + ".png")
            if not os.path.exists(png):
                base = png[:-4]
                ok = subprocess.run(["pdftoppm", "-r", str(a.dpi), "-gray", "-png", "-singlefile",
                                     "-f", "1", "-l", "1", src, base]).returncode == 0
                if not ok:
                    continue
            todo.append((r["page_id"], png))
        else:
            todo.append((r["page_id"], src))
    print(f"{len(todo)} pages to read", file=sys.stderr)

    cmd = [a.indecis, "ocr", "page", "-format", "html", "-det", a.det, "-model", a.model]
    if a.lm:
        cmd += ["-lm", a.lm]
    cmd += a.extra.split()
    ids = {path: pid for pid, path in todo}

    def read(batch):
        out = subprocess.run(cmd + [p for _, p in batch], capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr[-500:], file=sys.stderr)
            return 0
        n = 0
        for line in out.stdout.splitlines():
            d = json.loads(line)
            with open(os.path.join(pred_dir, ids[d["image"]] + ".json"), "w", encoding="utf-8") as f:
                json.dump({"pages": [{"html": d["html"]}]}, f, ensure_ascii=False)
            n += 1
        return n

    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    done = 0
    with ThreadPoolExecutor(a.procs) as ex:
        for n in ex.map(read, batches):
            done += n
            print(f"{done}/{len(todo)}", file=sys.stderr)
    with open(os.path.join(a.out, "settings.json"), "w") as f:
        json.dump({"dataset": ds, "manifest": "manifest.parquet"}, f)


if __name__ == "__main__":
    main()
