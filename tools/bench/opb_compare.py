"""indecis against other engines on the same Omni Parse Bench text tests,
from the per-test results the benchmark publishes (results.parquet).

    tools/oracle/.venv/bin/python tools/bench/opb_compare.py RUN-DIR DATASET-DIR

RUN-DIR is a run graded by `opb grade` (its unit_tests/details). The
published Indra OCR row reads the PDFs' text layer (its auto mode): the
fair comparison is on the image pages, where no text layer exists.
"""
import json, glob, sys
from collections import defaultdict
import pyarrow.parquet as pq
run, ds = sys.argv[1], sys.argv[2]
man = {r["page_id"]: r for r in pq.read_table(ds + "/manifest.parquet").to_pylist()}
ours = {}
for f in glob.glob(run + "/unit_tests/details/*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d["test_type"] in ("present", "order", "repeat"):
            ours[d["test_id"]] = d
systems = ["Indra OCR", "Tesseract", "Surya OCR 2", "PaddleOCR-VL-1.6", "olmOCR-2"]
t = pq.read_table(ds + "/results.parquet", columns=["system", "test_id", "result"]).to_pylist()
res = defaultdict(dict)
for r in t:
    if r["system"] in systems and r["test_id"] in ours:
        res[r["system"]][r["test_id"]] = r["result"] == "pass"
def slices(d):
    img = not man[d["page_id"]]["doc_path"].endswith(".pdf")
    s = ["all text"]
    if img: s.append("image pages")
    if "Latn" in d["tags"] and "math" not in d["tags"] and "handwriting" not in d["tags"] and "rotated" not in d["tags"]:
        s.append("Latin printed, no math")
        if img: s.append("Latin printed, image pages")
    if "fra" in d["tags"]: s.append("French")
    return s
agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for tid, d in ours.items():
    for s in slices(d):
        agg[s]["indecis"][0] += d["result"] == "pass"; agg[s]["indecis"][1] += 1
        for sy in systems:
            if tid in res[sy]:
                agg[s][sy][0] += res[sy][tid]; agg[s][sy][1] += 1
print(f"{'slice':30s} {'tests':>6s} " + " ".join(f"{x[:12]:>12s}" for x in ["indecis"] + systems))
for s, by in agg.items():
    n = by["indecis"][1]
    print(f"{s:30s} {n:6d} " + " ".join(f"{100*by[x][0]/max(1,by[x][1]):12.1f}" for x in ["indecis"] + systems))
