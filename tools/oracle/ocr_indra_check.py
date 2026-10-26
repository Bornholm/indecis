"""Local check of ocr_rec.py against Indra OCR's rec.onnx, and conversion of
its weights for the Go tests.

Indra's weights are under a modified OpenRAIL-M license: they are read here
as a reference only, written outside the repository, and never versioned.

    python ocr_indra_check.py --onnx-dir DIR --out ~/.cache/indecis/models/indra-rec

DIR holds rec.onnx and meta.json (huggingface.co/datalab-to/indra-ocr, v1.1.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file

from ocr_rec import Config, Recognizer, alone, greedy

FONT = "/usr/share/fonts/TTF/DejaVuSerif.ttf"
LINES = [
    "Le colis n'est pas arrivé, je veux parler à quelqu'un.",
    "Facture n° 2026-0412 : 1 234,56 € TTC",
    "The quick brown fox jumps over the lazy dog.",
    "ÉTAT DES LIEUX — 12 rue de l'Œuvre, 75004 Paris",
]


def weights(path: Path) -> dict[str, np.ndarray]:
    """rec.onnx's parameters under ocr_rec's names.  The exporter renamed the
    convolutions it fused with a batch norm (onnx::Conv_*): their node names
    give the module."""
    g = onnx.load(str(path)).graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    out = {k.removeprefix("rec."): v for k, v in init.items() if k.startswith("rec.")}
    for n in g.node:
        if n.op_type != "Conv" or not n.name.startswith("/rec/"):
            continue
        module = n.name.removeprefix("/rec/").removesuffix("/Conv").replace("/", ".")
        for suffix, name in zip(("weight", "bias"), n.input[1:]):
            out.setdefault(f"{module}.{suffix}", init[name])
    return out


def vocab(meta: Path) -> dict:
    v = json.loads(meta.read_text())["vocab"]
    return json.loads(v) if isinstance(v, str) else v


def render(text: str, c: Config) -> np.ndarray:
    """A clean synthetic line, cropped like Indra's crop_line (window of three
    line heights, 0.3 x height of margin left and right), as an ink map."""
    font = ImageFont.truetype(FONT, 48)
    x0, y0, x1, y1 = font.getbbox(text)
    h = y1 - y0
    pad_y = (c.window * 1.4 - 1) / 2 * h
    pad_x = 0.3 * h
    W, H = int(x1 - x0 + 2 * pad_x), int(h + 2 * pad_y)
    img = Image.new("L", (W, H), 255)
    ImageDraw.Draw(img).text((pad_x - x0, pad_y - y0), text, font=font, fill=0)
    out_h = c.window * c.height
    out_w = max(8, round(W * out_h / H))
    img = img.resize((out_w, out_h), Image.Resampling.BOX)
    return 1.0 - np.asarray(img, np.float32) / 255.0


def batch(lines: list[np.ndarray], multiple: int = 8):
    H = lines[0].shape[0]
    Wm = -(-max(x.shape[1] for x in lines) // multiple) * multiple
    x = np.zeros((len(lines), 1, H, Wm), np.float32)
    for i, l in enumerate(lines):
        x[i, 0, :, : l.shape[1]] = l
    return x, np.array([l.shape[1] for l in lines], np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    c = Config()
    m = Recognizer(c).eval()
    w = weights(a.onnx_dir / "rec.onnx")
    state = {k: torch.from_numpy(v.copy()) for k, v in w.items()}
    missing, unexpected = m.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (missing, unexpected)

    voc = vocab(a.onnx_dir / "meta.json")
    chars = voc["chars"]
    assert len(chars) + 1 == c.vocab, len(chars)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(a.onnx_dir / "rec.onnx"), so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    rand = [(rng.random((96, w)) > 0.85).astype(np.float32) for w in (40, 203, 517)]
    for name, lines in (("text", [render(t, c) for t in LINES]), ("random", rand)):
        x, widths = batch(lines)
        ids, conf, lengths, blank, topv, topi = sess.run(None, {"x": x, "widths": widths})
        with torch.no_grad():
            logits, tl = m(torch.from_numpy(x), torch.from_numpy(widths))
        lp = torch.log_softmax(logits, -1)
        tv, ti = lp.topk(8, -1)
        assert (tl.numpy() == lengths).all()
        d_topv = np.abs(tv.numpy() - topv).max()
        d_blank = np.abs(lp[..., 0].numpy() - blank).max()
        same_ids = (logits.argmax(-1).numpy() == ids).mean()
        print(f"{name}: max |Δ topv| {d_topv:.2e}, |Δ blank| {d_blank:.2e}, argmax agreement {same_ids:.4f}")
        assert d_topv < 1e-4 and d_blank < 1e-4, "PyTorch and onnxruntime disagree"
        if name == "text":
            for want, seq in zip(LINES, greedy(logits, tl)):
                print(f"  {''.join(chars[i - 1] for i in seq)!r}  (rendered {want!r})")

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.contiguous() for k, v in m.state_dict().items() if "num_batches" not in k}
        save_file(tensors, str(a.out / "model.safetensors"))
        (a.out / "config.json").write_text(c.to_json())
        (a.out / "vocab.json").write_text(json.dumps(voc, ensure_ascii=False))
        # Reference for the Go tests: each rendered line alone, its padded
        # ink map and the logits of its valid steps.
        xs, logits, widths = [], [], []
        with torch.no_grad():
            for t in LINES:
                ink = torch.from_numpy(render(t, c))
                x, w = alone(ink[None], ink.shape[1])
                l, n = m(x, w)
                xs.append(x[0, 0].numpy().ravel())
                logits.append(l[0, : int(n[0])].numpy().ravel())
                widths.append(int(w[0]))
        (a.out / "check_x.f32").write_bytes(np.concatenate(xs).astype(np.float32).tobytes())
        (a.out / "check_logits.f32").write_bytes(np.concatenate(logits).astype(np.float32).tobytes())
        (a.out / "check.json").write_text(json.dumps({"texts": LINES, "widths": widths}, ensure_ascii=False))
        print("weights written to", a.out)


if __name__ == "__main__":
    main()
