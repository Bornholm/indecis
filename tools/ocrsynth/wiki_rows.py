"""Paragraphs of a Wikipedia language, one per line, for ocrsynth -text,
fetched from the Hugging Face dataset viewer (datasets-server) by pages of
100 articles spread over the whole wiki: a few thousand articles of each
language without downloading its parquet shards.

    python3 tools/ocrsynth/wiki_rows.py vi wiki_vi.txt [paragraphs] [pages]

Wikipedia's text is under CC BY-SA 4.0: it is rendered into training
images, never shipped.
"""

import json
import random
import sys
import time
import urllib.request

API = "https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.{lang}&split=train&offset={off}&length=100"


def fetch(lang, off):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(API.format(lang=lang, off=off), timeout=120) as r:
                return json.load(r)
        except Exception as e:  # rate limits and timeouts: wait and retry
            print(f"{lang} {off}: {e}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    return None


def main():
    lang, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30_000
    pages = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    first = fetch(lang, 0)
    if first is None:
        sys.exit(f"{lang}: no answer")
    total = first["num_rows_total"]
    rng = random.Random(lang)
    offsets = sorted(rng.sample(range(0, max(1, total - 100)), min(pages, max(1, total // 100))))
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for i, off in enumerate(offsets):
            d = first if i == 0 and off == 0 else fetch(lang, off)
            if d is None:
                continue
            for row in d["rows"]:
                for p in row["row"]["text"].split("\n"):
                    p = " ".join(p.split())
                    # Running text, not headings, lists or tables.
                    if len(p) >= 80 and not p.startswith(("|", "{", "*", "#")):
                        out.write(p + "\n")
                        n += 1
                        if n >= limit:
                            print(f"{lang}: {n} paragraphs", file=sys.stderr)
                            return
    print(f"{lang}: {n} paragraphs", file=sys.stderr)


if __name__ == "__main__":
    main()
