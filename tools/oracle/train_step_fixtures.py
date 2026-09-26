"""Fixtures de parité d'un pas d'entraînement : gradients, écrêtage et pas
AdamW selon PyTorch.

Perte : L = Σ_b ⟨moyenne des états cachés de b, w_b⟩, w tiré au hasard.
Chaque séquence passe seule (pas de padding), les gradients s'additionnent.

    tools/oracle/.venv/bin/python tools/oracle/train_step_fixtures.py \
        --model ~/.cache/indecis/models/bekko-embedding-v1-a8m --out testdata/bekko
"""
import argparse, json, os, zlib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import forward_fixtures as ff

TEXTS = [ff.TEXTS[1], ff.TEXTS[3], ff.LONG]
LR, WD, CLIP = 1e-3, 0.01, 1.0
SAMPLES = 64

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model_dir = os.path.expanduser(args.model)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    H = model.config.hidden_size
    w = torch.from_numpy(np.random.default_rng(0).standard_normal((len(TEXTS), H)).astype(np.float32))

    ids = [tok(t, return_tensors="pt")["input_ids"] for t in TEXTS]
    loss = sum((model(input_ids=x).last_hidden_state[0].mean(0) * w[i]).sum() for i, x in enumerate(ids))
    loss.backward()

    used = sorted({int(i) for x in ids for i in x[0]})
    params = dict(model.named_parameters())
    before = {n: p.detach().clone() for n, p in params.items()}

    def pick(name, p):
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        if name == "embeddings.tok_embeddings.weight":
            rows = rng.choice(used, SAMPLES)
            cols = rng.integers(0, H, SAMPLES)
            return (rows * H + cols).tolist()
        return rng.integers(0, p.numel(), SAMPLES).tolist()

    out = {"texts": TEXTS, "w": w.flatten().tolist(), "loss": float(loss), "params": {}}
    for n, p in params.items():
        idx = pick(n, p)
        g = p.grad.flatten()
        out["params"][n] = {"idx": idx, "grad": g[idx].tolist(), "grad_norm": float(g.norm())}

    out["total_norm"] = float(torch.nn.utils.clip_grad_norm_(list(params.values()), CLIP))

    emb = params["embeddings.tok_embeddings.weight"]
    decay = [p for n, p in params.items() if p.dim() == 2 and p is not emb]
    no_decay = [p for n, p in params.items() if p.dim() == 1]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": WD},
        {"params": no_decay + [emb], "weight_decay": 0.0},
    ], lr=LR, betas=(0.9, 0.999), eps=1e-8)
    opt.step()

    for n, p in params.items():
        e = out["params"][n]
        flat = p.detach().flatten()
        e["after"] = flat[e["idx"]].tolist()
        e["delta_norm"] = float((p.detach() - before[n]).norm())
    out.update(lr=LR, weight_decay=WD, clip=CLIP)

    with open(os.path.join(args.out, "train_step.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"loss={out['loss']:.6f} total_norm={out['total_norm']:.6f} params={len(params)} used_rows={len(used)}")

if __name__ == "__main__":
    main()
