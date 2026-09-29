"""SigLIP 2 parity fixtures: image preprocessing, image and text
embeddings, and image-text logits according to transformers.

    tools/oracle/.venv/bin/python tools/oracle/siglip_fixtures.py \
        --model ~/.cache/indecis/models/siglip2-base-patch32-256 \
        --out testdata/siglip2

The test images are synthetic (no third-party content): a gradient with
shapes, and noise, at sizes that exercise downscaling and upscaling.
"""
import argparse, json, os

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

TEXTS = [
    "a red circle on a blue background",
    "random colored noise",
    "a photo of a cat",
    "un cercle rouge sur fond bleu",
    "Une photo d'un chat qui dort sur un canapé, très longue description pour dépasser éventuellement la limite de soixante-quatre tokens imposée par le modèle SigLIP, avec encore quelques mots en plus pour être sûr.",
]


def images():
    w, h = 320, 240
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    grad[..., 2] = np.linspace(80, 255, w, dtype=np.uint8)[None, :]
    grad[..., 1] = np.linspace(0, 120, h, dtype=np.uint8)[:, None]
    a = Image.fromarray(grad)
    d = ImageDraw.Draw(a)
    d.ellipse((110, 60, 210, 160), fill=(220, 30, 30))
    d.rectangle((20, 180, 90, 230), fill=(250, 250, 250))
    rng = np.random.default_rng(1)
    b = Image.fromarray(rng.integers(0, 256, (97, 131, 3), dtype=np.uint8))
    return {"shapes": a, "noise": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    cases = {"images": [], "texts": []}
    pix = []
    for name, im in images().items():
        im.save(f"{args.out}/{name}.png")
        # The resized uint8 image, as PIL produces it inside the processor.
        size = proc.image_processor.size
        resized = np.asarray(im.convert("RGB").resize((size["width"], size["height"]), resample=Image.BILINEAR))
        resized.tofile(f"{args.out}/{name}_resized.u8")
        pv = proc(images=im, return_tensors="pt")["pixel_values"]
        # The processor's pixels are exactly (resized/255 - 0.5)/0.5.
        ref = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        assert torch.allclose(pv[0], (ref - 0.5) / 0.5, atol=1e-6)
        pix.append(pv)
        cases["images"].append({"name": name})
    pixel_values = torch.cat(pix)

    tok = proc.tokenizer(TEXTS, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        img = model.get_image_features(pixel_values=pixel_values)
        txt = model.get_text_features(input_ids=tok["input_ids"])
        out = model(pixel_values=pixel_values, input_ids=tok["input_ids"])
    img = getattr(img, "pooler_output", img)
    txt = getattr(txt, "pooler_output", txt)
    for i, c in enumerate(cases["images"]):
        c["embedding"] = img[i].tolist()
    for i, t in enumerate(TEXTS):
        cases["texts"].append({"text": t, "ids": tok["input_ids"][i].tolist(), "embedding": txt[i].tolist()})
    cases["logits_per_image"] = out.logits_per_image.tolist()
    cases["logit_scale"] = model.logit_scale.item()
    cases["logit_bias"] = model.logit_bias.item()
    with open(f"{args.out}/cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    print(f"{len(cases['images'])} images, {len(TEXTS)} texts -> {args.out}")
    print("logits", out.logits_per_image)


if __name__ == "__main__":
    main()
