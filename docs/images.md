# Deciding on images

indecis also decides on images, with a SigLIP model and no training: the options of a question are described in text, in English or French, and each is scored against the image. This is open mode (see [open-categories.md](open-categories.md)), with an image instead of a text.

The model is [google/siglip2-base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256): an 86M-parameter image encoder and a multilingual text encoder, under the Apache-2.0 license.

```bash
M=~/.cache/indecis/models/siglip2-base-patch32-256
mkdir -p $M
for f in config.json model.safetensors tokenizer.json; do
  curl -sL -o $M/$f https://huggingface.co/google/siglip2-base-patch32-256/resolve/main/$f
done
```

The file weighs 1.5 GB, most of it the text embedding table (256,000 tokens × 768).

## From Go

```go
m, _ := vision.Load(dir, vision.WithInt8())
a, _ := m.ChooseNearest(ctx, []indecis.Candidate{
    {Name: "truck", Description: "a photo of a garbage truck"},
    {Name: "car", Description: "a photo of a car"},
}, img)
a.Choice // "truck"
```

- `ChooseNearest` picks the option that best describes the image.
- `DecideOpen` answers `indecis.OpenQuestion` values: choice, score, yes/no.
- `Match` returns the model's own probability that a text describes the image.
- `EmbedImage`, `EmbedText` and `Logit` give the embeddings and the score.

Describe each option as a caption: "a photo of a garbage truck" rather than "truck". The text encoder reads the description, or the name when there is none; the question's instructions are not read for a choice.

## Trained questions

When describing the options is not enough, or a question depends on where something is in the image, train a head on examples. The encoder stays frozen; the head reads the features of each of the 64 patches (32×32 pixels) at its position, and their maxima over the image for questions about anywhere.

```bash
bin/indecis train-vision -backbone $M -schema schema.json \
    -train images.jsonl -test test.jsonl -out model
```

Each line of the JSONL is `{"image": "frames/000123.png", "labels": {"fire": true, "turn": "left"}}`, with paths relative to the file, and the schema is the one of a text model (see [concepts.md](concepts.md)). Encoding takes about 65 ms per image and core (`-workers`); training the head, a few seconds to a minute. The output directory holds `vision.json` and a head of about 100 KB; `vision.Load` and `indecis-serve` read it like an encoder, and answer its questions with the head, any other in open mode, with one pass of the encoder.

`-layer n` trains the head on the patches after n layers instead of the last: when a model has only learned questions, the encoder then stops there. In indecis-vizdoom, layer 8 of 12 kept the accuracy, cut the latency by 29% and raised the score in play.

On Imagenette (1,000 training images, 300 test), the head gets 99.3%, against 99.0% zero-shot. In [indecis-vizdoom](../../indecis-vizdoom), a head trained on 13,342 Doom frames (half of them mirrored), reading layer 8, plays from pixels alone at 97% of the scripted policy's score, deciding in 52 ms.

## Behind the decision API

`indecis-serve` recognizes a SigLIP model by its `config.json` and serves it as an image model. The state is the image, as a data URL or as `{"image": ...}` (PNG, JPEG or GIF, 64 megapixels at most); every question is open.

```bash
bin/indecis-serve -model images=$M
IMG=$(base64 -w0 photo.jpg)
curl -s localhost:8080/api/alpha/decisions -d '{
  "state": "data:image/jpeg;base64,'$IMG'",
  "questions": {
    "vehicle": {"type": "choice", "instructions": "Vehicle",
      "criteria": {"truck": "une photo d'\''un camion poubelle", "car": "une photo d'\''une voiture"}},
    "outdoors": {"type": "noul", "instructions": "Outdoors",
      "criteria": {"true": "une photo prise en extérieur", "false": "une photo prise en intérieur"}}
  }
}'
```

## Measured quality

Zero-shot, on 500 images of the validation split of [Imagenette](https://github.com/fastai/imagenette) (10 ImageNet classes, 50 per class), each class described by a caption:

| Captions | float32 | int8 |
| --- | --- | --- |
| "a photo of a tench." | 99.0% | 99.0% |
| "une photo de tanche." | 98.8% | 98.8% |
| the class name only | 99.0% | 99.2% |

transformers gets 99.2% on the same images and captions; the difference is one image, decoded slightly differently by Go's JPEG decoder. Imagenette's classes are far apart: expect less on close categories. `go run ./tools/imgeval` repeats the measurement.

**Yes/no questions: describe both criteria.** SigLIP ranks images well on a property (AUC 0.97 to 1.00 for "a photo of an animal", "a vehicle", "a musical instrument"), but its own probability (`Match`) is set for precise captions and stays under 0.5 for most positives. `DecideOpen` opposes the two criteria of a noul question instead. With only instructions, they face a fixed anchor, "Something else": 86 to 97% right at 0.5 on those properties. With a described negation ("a photo of an object"), 92 to 98%. On a photo of a garbage truck, "une photo prise en extérieur" alone gives 0.31; against "une photo prise en intérieur", 0.89.

## Speed and memory

On a laptop (Core Ultra 7 265U):

| | float32, one core | int8, one core | int8, two performance cores |
| --- | --- | --- | --- |
| an image (64 patches of 32 pixels) | 148 ms | 65 to 73 ms | 48 ms |
| a text (64 tokens) | 141 ms | 52 ms | |

Text embeddings are cached, so an option costs once; an image costs every time. One image uses one core by default, leaving the others to simultaneous requests; `vision.WithThreads(2)` (or `indecis-serve -threads 2`) spreads it. On a hybrid processor, go no further than the performance cores: on all fourteen cores, an image takes 160 ms, the efficiency cores holding the others up.

**int8 and massive activations.** In a few layers, some input channels of the second MLP product (`fc2`) reach a hundred times the others: up to 1,479 against a median of 3.4 in layer 9. Plain per-token int8 flattens the rest of each row. `linalg.MatMul8Outliers` finds those channels on each call, takes them out of the int8 product and multiplies them in float32 (the LLM.int8() decomposition), with no calibration. On 300 Imagenette images, the logits move by 0.21 on average from float32 (0.17 with `fc2` kept in float32), for the same accuracy, 99.7%, and 37% less time.

## Parity with the reference implementation

| Stage | Test | Result |
| --- | --- | --- |
| Tokenizer (original Gemma pipeline, no `<bos>`) | 3,053 texts | identical ids |
| Resize (PIL's bilinear filter) | downscale and upscale | identical bytes |
| Image and text embeddings, float32 | 2 images, 5 texts | max relative difference 2.4·10⁻⁶ |
| Logits, int8 | 10 pairs | within 0.45, same ranking |

Fixtures come from `tools/oracle/siglip_fixtures.py`, with synthetic images.
