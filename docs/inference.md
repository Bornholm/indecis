# Inference speed and memory

indecis runs on CPU, in pure Go, without cgo. On one laptop core (Core Ultra 7 265U), the prompt-injection detector judges a 15-token sentence in 1.1 ms and a 256-token text in 15 ms. A model ready to serve takes 13 to 15 MB of its own memory, and loads in a few milliseconds with its `tokenizer.bin`.

## Measurements

Prompt-injection detector, a8m, one core, measured with `go run ./tools/infbench -model … -int8`:

| | Training forward | Inference path, float32 | int8 and portable SIMD | int8, compacted model | AVX2 activation kernels, `tokenizer.bin` |
| --- | --- | --- | --- | --- | --- |
| 15 tokens | 10.7 ms | 4.9 ms | 1.5 ms | 1.5 ms | 1.1 ms |
| 256 tokens | 79.8 ms | 57.4 ms | 23 ms | 23 ms | 14.7 ms |
| Own memory, model ready | 486 MB | 53 MB | 30 MB | 18 MB | 13 MB |
| Peak memory | 1,037 MB | 222 MB | 82 MB | 51 MB | 38 MB |
| File | 289 MB | 289 MB | 289 MB | 58.5 MB | 58.5 MB + 4.4 MB |

The last column is the compacted model, measured on a performance core (`taskset -c 2`); on the same core, the previous code takes 1.6 ms and 19.4 ms. The full model (256,000 tokens) computes as fast and loads in 27 ms with its `tokenizer.bin`, against 470 ms by parsing its `tokenizer.json`; its own memory once ready drops from 28 to 15 MB, its peak from 76 to 45 MB. Spread over all cores, an 89-token text takes 2.5 ms instead of 5.5 ms on one core, a 34-token one 2.0 ms instead of 2.2. A 256-token text takes about 5 ms: there, the wait for the slowest core dominates, the low-power cores computing five times slower than the performance cores.

On the 636-example hand-reviewed reference, the AUC stays at 0.972, and 0.970 for the compacted model. The AVX2 kernels change no metric of the reference.

## Settings

`indecis check` reports what speeds up computation on the current machine: portable SIMD in hardware or emulated, vector width, AVX2 and FMA, AVX-VNNI. It exits with status 1 when SIMD is missing or emulated. From Go: `indecis.DetectHardware()`.

| Option | Effect |
| --- | --- |
| `WithInt8()` | layers computed in int8 when the CPU has AVX-VNNI (Intel since Alder Lake, AMD since Zen 4); no effect elsewhere |
| `WithThreads(n)` | maximum number of cores (0: all) |
| `WithMaxLen(n)` | tokens read per text, 256 by default |
| `WithEmbedCache(n)` | keeps the embeddings of open-mode options |
| `WithBatching(w, r)` | groups simultaneous requests (see below) |

Under 32 tokens, a computation stays on one core: waking the others would cost more than it saves. Above that, it spreads over the allowed cores, in slices of rows that shrink with the text, so that a slow core does not hold up the others for long. A computation takes the cores the others leave free, each computation in progress occupying at least one: on a saturated server, each stays on one core. `WithThreads(0)` therefore suits short and long texts alike.

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

Every `indecis compact` also writes a `tokenizer.bin` next to the `tokenizer.json`: the tables of the tokenizer as they sit in memory, mapped at load time instead of parsed (13.7 MB for 256,000 tokens, 4.4 MB for the pruned vocabulary). `indecis compact -model dir`, without `-corpus` or `-int8-embeddings`, writes only that file; from Go, `tokenizer.WriteBinary`. The file is used only if it was written from the `tokenizer.json` beside it (SHA-256 digest): a JSON changed afterwards is parsed again.

For a known domain, prune on the training data. For a model that will see unforeseen texts, keep the whole table and use `-int8-embeddings` only. Never put evaluation sets in the pruning corpus.

## What makes inference fast

- **A dedicated inference path**: nothing is kept for backpropagation, buffers are reused, and the weights are prepared once for the matrix products.
- **int8**: weights quantized per channel, activations per token, exact integer sums. The micro-kernel uses the AVX-VNNI `VPDPBUSD` instruction, in its VEX form that the Go assembler does not know: `internal/linalg/gen_vnni.py` writes it byte by byte.
- **Go 1.27 portable SIMD** for everything around the products: quantization, rescaling, GELU, normalization, softmax. Each operation has a scalar fallback with the same contract, and none is written in assembly.
- **A memory-mapped embedding table**: only the pages of the tokens actually met are read.
- **Few passes around the products**, in AVX2 assembly where portable SIMD lacks registers: the activations are quantized in one pass (maximum, then rounding narrowed 32 values at a time), the gated GELU keeps its 17 constants in memory operands rather than spilling them, and the last rows of a product, when the text length is not a multiple of 6, are rescaled in the kernel too.
- **Lean loading**: the tokenizer is mapped from its `tokenizer.bin` (or streamed from its JSON) and shared between models with identical tokenizer files, and the layer matrices are quantized one at a time.

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

`go run ./tools/infbench -model … -int8 -threads 0 -clients n` measures the throughput of n simultaneous clients, each sending one length of text, then all lengths in turn (15 to 256 tokens). On this mixed load, with 4 clients, throughput rises by 19% (604 to 719 requests/s) and the latency of 34- and 89-token texts halves, while 256-token texts take 15% longer (10.9 ms instead of 9.5): they no longer take every core. With 8 and 16 clients, throughput does not change; 89-token texts wait 22% less with 8 clients, 256-token texts 8 to 14% more.
