"""Paragraphs of a Wikipedia parquet shard (huggingface.co/datasets/
wikimedia/wikipedia), one per line, for ocrsynth -text.

    tools/oracle/.venv/bin/python tools/ocrsynth/wiki_text.py fr0.parquet fr.txt

Wikipedia's text is under CC BY-SA 4.0: it is rendered into training
images, never shipped.
"""

import sys

import pyarrow.parquet as pq


def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for batch in pq.ParquetFile(src).iter_batches(columns=["text"], batch_size=1000):
            for text in batch.column(0).to_pylist():
                for p in text.split("\n"):
                    p = " ".join(p.split())
                    # Running text, not headings, lists or tables.
                    if len(p) >= 80 and not p.startswith(("|", "{", "*", "#")):
                        out.write(p + "\n")
                        n += 1
                        if n >= limit:
                            return


if __name__ == "__main__":
    main()
