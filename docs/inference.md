# Inference speed and memory

indecis runs on CPU, in pure Go, without cgo. On one laptop core (Core Ultra 7 265U), the prompt-injection detector judges a 15-token sentence in 1.5 ms and a 256-token text in 23 ms. A model ready to serve takes 18 to 30 MB of its own memory.

## Measurements

Prompt-injection detector, a8m, one core, measured with `go run ./tools/infbench -model … -int8`:

| | Training forward | Inference path, float32 | int8 and portable SIMD | int8, compacted model |
| --- | --- | --- | --- | --- |
| 15 tokens | 10.7 ms | 4.9 ms | 1.5 ms | 1.5 ms |
| 256 tokens | 79.8 ms | 57.4 ms | 23 ms | 23 ms |
| Own memory, model ready | 486 MB | 53 MB | 30 MB | 18 MB |
| Peak memory | 1,037 MB | 222 MB | 82 MB | 51 MB |
| File | 289 MB | 289 MB | 289 MB | 58.5 MB |

On the 636-example hand-reviewed reference, the AUC stays at 0.972, and 0.970 for the compacted model.

## Settings

`indecis check` reports what speeds up computation on the current machine: portable SIMD in hardware or emulated, vector width, AVX2 and FMA, AVX-VNNI. It exits with status 1 when SIMD is missing or emulated. From Go: `indecis.DetectHardware()`.

| Option | Effect |
| --- | --- |
| `WithInt8()` | layers computed in int8 when the CPU has AVX-VNNI (Intel since Alder Lake, AMD since Zen 4); no effect elsewhere |
| `WithThreads(n)` | maximum number of cores (0: all) |
| `WithMaxLen(n)` | tokens read per text, 256 by default |
| `WithEmbedCache(n)` | keeps the embeddings of open-mode options |
| `WithBatching(w, r)` | groups simultaneous requests (see below) |

Under 1,024 tokens, a computation stays on one core, the fastest for a sentence: on a hybrid CPU, spreading it would send it to the efficiency cores. Above that, it spreads over the allowed cores. `WithThreads(0)` therefore suits short and long texts alike.

int8 does not change the decisions on our references (92.6% accuracy in int8, 92.5% in float32 for the prompt-injection detector). Check it on your own model: `indecis eval` computes in int8 by default, and in float32 with `-int8=false`.

## Shrinking a model

The embedding table (256,000 tokens × 384 dimensions) holds 93% of the weights. `indecis compact` offers two reductions:

- **`-int8-embeddings`** stores the table in int8, with one scale per row. The file size is halved with no measured loss on our references, open mode included. Hugging Face tools cannot read this format.
- **`-corpus`** removes from the vocabulary the tokens that a representative corpus never uses. A text covered by the corpus is tokenized exactly as before; a word missing from the corpus is split into smaller pieces, which the model never saw in training.

Pruning is a decision to make knowingly:

| Model | Vocabulary | File | Measured effect |
| --- | --- | --- | --- |
| prompt-injection detector, pruned on its training data | 256,000 → 78,890 | 289 → 160 MB | none (AUC 0.972 → 0.970) |
| same model, pruned, with an int8 table | 78,890 | 58.5 MB | none |
| email classification in open mode, unforeseen texts | 256,000 → 119,480 | 260 → 160 MB | Enron 41.9 → 38.4%, ASN letters 23.8 → 18.4% |

For a known domain, prune on the training data. For a model that will see unforeseen texts, keep the whole table and use `-int8-embeddings` only. Never put evaluation sets in the pruning corpus.

## What makes inference fast

- **A dedicated inference path**: nothing is kept for backpropagation, buffers are reused, and the weights are prepared once for the matrix products.
- **int8**: weights quantized per channel, activations per token, exact integer sums. The micro-kernel uses the AVX-VNNI `VPDPBUSD` instruction, in its VEX form that the Go assembler does not know: `internal/linalg/gen_vnni.py` writes it byte by byte.
- **Go 1.27 portable SIMD** for everything around the products: quantization, rescaling, GELU, normalization, softmax. Each operation has a scalar fallback with the same contract, and none is written in assembly.
- **A memory-mapped embedding table**: only the pages of the tokens actually met are read.
- **Lean loading**: the tokenizer is streamed and shared between models with identical tokenizer files, and the layer matrices are quantized one at a time.

## Long texts

Attention is computed in blocks, with a softmax updated block after block: memory grows with the length, not with its square. The local layers (128-token window) only visit the blocks of their window.

| Tokens | a8m, one core | a8m, all cores | a25m, all cores |
| --- | --- | --- | --- |
| 256 | 25 ms | 24 ms | 75 ms |
| 1,024 | 149 ms | 82 ms | 268 ms |
| 4,096 | 1.4 s | 0.65 s | 1.9 s |

The global layers (one in three) stay quadratic in compute. Our models were fine-tuned on 256 tokens; beyond that, their quality has not been measured.

## Server under load

Each request holds its own buffers, about 10 MB for 256 tokens. Memory therefore levels off according to the number of simultaneous requests, which `indecis-serve -max-concurrent` and `-memory-limit` bound. With two models served and 16 parallel clients: 394 MB at most with the default settings, 246 MB with `-max-concurrent 8 -memory-limit 256`, for 12% less throughput.

Grouping simultaneous requests into batches (`WithBatching`, `indecis-serve -batching`) brings little on CPU: 3 to 8% more throughput on one-sentence requests, nothing on 256-token requests, and more memory. It is off by default.
