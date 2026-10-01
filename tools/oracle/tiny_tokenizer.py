"""A small tokenizer.json with SigLIP 2's pipeline (raw Gemma: Replace
normalizer, Split pre-tokenizer, no <bos>), cut from the real one, so that
CI tests that pipeline without the 1.5 GB checkpoint.

    tools/oracle/.venv/bin/python tools/oracle/tiny_tokenizer.py \
        --model ~/.cache/indecis/models/siglip2-base-patch32-256 \
        --out testdata/siglip2-tiny

Kept: the tokens with the lowest ids (special, byte fallback, the first
merges), the single characters of Latin-1, and the merges among them; ids
are renumbered densely. Other characters go through byte fallback.
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=3000)
    args = ap.parse_args()

    j = json.load(open(os.path.join(args.model, "tokenizer.json"), encoding="utf-8"))
    vocab = j["model"]["vocab"]
    latin = lambda s: len(s) == 1 and (ord(s) < 0x100 or s == "▁")
    kept = sorted((i, s) for s, i in vocab.items() if i < args.keep or latin(s))
    remap = {old: new for new, (old, _) in enumerate(kept)}
    j["model"]["vocab"] = {s: remap[i] for i, s in kept}
    merges = []
    for m in j["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        if a in j["model"]["vocab"] and b in j["model"]["vocab"] and a + b in j["model"]["vocab"]:
            merges.append(m)
    j["model"]["merges"] = merges
    j["added_tokens"] = [dict(t, id=remap[t["id"]]) for t in j["added_tokens"] if t["id"] in remap]
    for tok in j["post_processor"].get("special_tokens", {}).values():
        tok["ids"] = [remap[i] for i in tok["ids"]]
    for key in ("padding", "truncation"):
        j[key] = None
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(kept)} tokens, {len(merges)} merges -> {args.out}")

if __name__ == "__main__":
    main()
