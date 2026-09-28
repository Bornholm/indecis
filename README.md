# indecis

indecis builds small decision models in Go: a pretrained text encoder, fully fine-tuned on your examples, that answers typed questions with calibrated probabilities instead of generated text.

```bash
bin/indecis synth -templates examples/guide/templates -n 900 -out data.jsonl
bin/indecis train -backbone $BEKKO -schema examples/guide/schema.json -train data.jsonl -out model
echo "Mon colis n'est pas arrivé, je veux parler à quelqu'un." | bin/indecis predict -model model
```

```json
{"answers": {"sujet": {"choice": "livraison", …}, "urgence": {"score": 2.0, …}, "humain": {"p": 1.0, …}}, …}
```

There are three question types: yes/no (`noul`), one option among several (`choice`), and a level on an ordered scale (`score`). One pass of the model answers all of them.

## In numbers

Measured on a laptop (Core Ultra 7 265U) with the bekko-embedding-v1-a8m backbone (7.7M parameters, multilingual):

- **1.5 ms** to judge a 15-token sentence, **23 ms** for 256 tokens, on a single core;
- **18 to 30 MB** of memory for a model ready to serve, and a **58 to 290 MB** file;
- **half an hour** to train on 20,000 examples;
- no GPU, no cgo, no dependency outside the Go standard library.

## Documentation

- [Creating a model in five steps](docs/creating-a-model.md): schema, data, training, evaluation, deployment.
- [Concepts](docs/concepts.md): question types, calibration, (context, text) pairs, fixed answers or open options.
- [Producing data](docs/data.md): templates, labeling by LLMs, consensus.
- [Open categories](docs/open-categories.md): classify among a list that changes with every request.
- [Serving a model](docs/serving.md): HTTP server compatible with TypeSafe and OpenRouter, provider for genai.
- [Inference speed and memory](docs/inference.md): int8, SIMD, shrinking a model, long texts.
- [Architecture](docs/architecture.md): packages, parity with PyTorch, tests.

## Status

Experimental: the API may still change. Requires Go 1.27. The experimental SIMD support (`GOEXPERIMENT=simd`) makes it fast; indecis also runs without it. License: MIT.
