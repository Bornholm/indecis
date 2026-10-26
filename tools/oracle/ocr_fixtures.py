"""Parity fixtures of the OCR line recognizer: a tiny model with random
weights, its logits, and a training step (CTC loss, gradients, clipping,
AdamW) according to PyTorch.

    tools/oracle/.venv/bin/python tools/oracle/ocr_fixtures.py --out testdata/ocr-tiny

Writes, for each stem normalization (batch, layer), a folder holding
config.json, model.safetensors, logits.f32 (the valid steps of each line,
concatenated: [sum of lengths, vocab]) and, for "layer"
(the trainable variant), grads.safetensors and after.safetensors.  The
inputs and targets are in cases.json at the top.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ocr_rec import Config, Recognizer, alone

WIDTHS = [37, 64, 21]  # true widths: one line ends mid-step, one is short
TARGETS = [[3, 3, 5], [1, 2, 3, 4, 5, 6, 7, 1], [9, 2]]
LR, WD, CLIP = 1e-3, 0.01, 1.0


def tiny(norm: str) -> Config:
    return Config(
        height=8, window=3, stride=4, dim=16, expand=24, se=8, blocks=2, kernel=5,
        ctx_channels=8, ctx_dim=12, ctx_heads=3, vocab=11, norm=norm,
    )


def randomize(m: torch.nn.Module, seed: int):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.5 if p.dim() > 1 else 0.2))
        for name, b in m.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.3)
            elif name.endswith("running_var"):
                b.copy_(torch.rand(b.shape, generator=g) + 0.5)


def tensors(m: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().contiguous() for k, v in m.state_dict().items() if "num_batches" not in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    c = tiny("batch")
    rng = np.random.default_rng(1)
    H = c.window * c.height
    Wm = -(-max(WIDTHS) // 8) * 8
    x = np.zeros((len(WIDTHS), 1, H, Wm), np.float32)
    for i, w in enumerate(WIDTHS):
        x[i, 0, :, :w] = rng.random((H, w)).astype(np.float32)
    (a.out / "x.f32").write_bytes(x.tobytes())
    xt = torch.from_numpy(x)

    cases = {"shape": list(x.shape), "widths": WIDTHS, "targets": TARGETS,
             "lr": LR, "weight_decay": WD, "clip": CLIP}
    for norm in ("batch", "layer"):
        c = tiny(norm)
        d = a.out / norm
        d.mkdir(exist_ok=True)
        m = Recognizer(c).eval()
        randomize(m, 7)
        (d / "config.json").write_text(c.to_json())
        save_file(tensors(m), str(d / "model.safetensors"))
        # Each line alone, as the Go implementation reads it (see ocr_rec.alone).
        outs = [m(*alone(xt[i], w)) for i, w in enumerate(WIDTHS)]
        logits = [o[0][0, : int(o[1][0])] for o in outs]
        (d / "logits.f32").write_bytes(torch.cat(logits).detach().numpy().astype(np.float32).tobytes())
        cases[norm] = {"lengths": [len(l) for l in logits]}
        if norm != "layer":
            continue

        loss = sum(
            F.ctc_loss(F.log_softmax(l, -1)[:, None], torch.tensor([t]), torch.tensor([len(l)]),
                       torch.tensor([len(t)]), blank=0, reduction="sum")
            for l, t in zip(logits, TARGETS)
        )
        loss.backward()
        params = dict(m.named_parameters())
        save_file({k: p.grad.contiguous() for k, p in params.items()}, str(d / "grads.safetensors"))
        cases[norm]["loss"] = float(loss.detach())
        cases[norm]["total_norm"] = float(torch.nn.utils.clip_grad_norm_(list(params.values()), CLIP))
        # Weight decay on matrices and convolution kernels, not on biases,
        # norms and the row prior.
        decay = [p for n, p in params.items() if p.dim() > 1 and "row_prior" not in n]
        rest = [p for n, p in params.items() if not (p.dim() > 1 and "row_prior" not in n)]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                                 {"params": rest, "weight_decay": 0.0}],
                                lr=LR, betas=(0.9, 0.999), eps=1e-8)
        opt.step()
        save_file(tensors(m), str(d / "after.safetensors"))

    (a.out / "cases.json").write_text(json.dumps(cases, indent=1))
    print(json.dumps({k: v for k, v in cases.items() if k in ("batch", "layer")}))


if __name__ == "__main__":
    main()
