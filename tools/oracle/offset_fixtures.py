"""Offset parity fixtures: for each single text of a tokenizer fixture
file, the byte span of every token according to Hugging Face's
`tokenizers` library.

    tools/oracle/.venv/bin/python tools/oracle/offset_fixtures.py \
        --model ~/.cache/indecis/models/bekko-embedding-v1-a8m \
        --cases testdata/bekko/tokenizer_cases.jsonl \
        --out testdata/bekko/offset_cases.jsonl

The library reports offsets in characters; they are converted to UTF-8
bytes, the unit of Go strings.
"""
import argparse, json
from tokenizers import Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    tok.no_padding()
    tok.no_truncation()
    n = 0
    with open(args.cases, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8") as f:
        for line in src:
            case = json.loads(line)
            if case.get("pair"):
                continue
            t = case["text"]
            # byte offset of each character, and of the end
            pos = [0]
            for ch in t:
                pos.append(pos[-1] + len(ch.encode("utf-8")))
            enc = tok.encode(t, add_special_tokens=True)
            offs = [[pos[a], pos[b]] for a, b in enc.offsets]
            f.write(json.dumps({"text": t, "ids": enc.ids, "offsets": offs}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} cases -> {args.out}")

if __name__ == "__main__":
    main()
