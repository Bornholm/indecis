# Text embeddings

indecis computes sentence embeddings with the same encoder and the same inference path as its decision models (int8, SIMD, memory-mapped weights). A query and the documents it should find are embedded on their own and compared by their dot product, their cosine since the vectors have norm 1. This is the dense half of a search engine such as Amoxtli.

## Models

Any sentence-transformers directory with a ModernBERT encoder, mean or CLS pooling and a Gemma tokenizer loads as is:

| Model | Parameters outside the table | Dimension | License | Notes |
| --- | --- | --- | --- | --- |
| [bekko-embedding-v1-a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m) | 7.7M | 384 | MIT | the default backbone, fastest |
| [bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m) | ~25M | 384 | MIT | 13 layers |
| [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) | 110M | 768 (Matryoshka down to 128) | Apache 2.0 | 22 layers, CLS pooling, 32k positions |

The pooling comes from `1_Pooling/config.json`, the query and document prompts from `config_sentence_transformers.json` (none of these three models uses prompts). Models built differently (Qwen3-Embedding, byte-level tokenizers, SiLU activations) are refused at load time.

## Quality

Dense retrieval over the whole corpus (exact cosine), nDCG@10, at most 512 tokens per text. *scifact-fr* asks the SciFact questions in French over the English corpus; *squad-fr* finds the French Wikipedia passage of each question (761 passages, 3,835 questions). *Hybrid* fuses the ranking with BM25 (reciprocal rank fusion, equal weights), as Amoxtli does.

| Model (indecis, int8) | scifact | scifact-fr | nfcorpus | squad-fr | hybrid scifact | hybrid nfcorpus | hybrid squad-fr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 alone | 0.660 | 0.315 | 0.306 | 0.763 | | | |
| bekko a8m | 0.698 | 0.595 | 0.334 | 0.791 | 0.708 | 0.342 | 0.812 |
| bekko a25m | 0.715 | 0.591 | 0.347 | 0.799 | 0.712 | 0.342 | 0.819 |
| granite-311m | 0.713 | 0.655 | 0.309 | 0.759 | 0.717 | 0.334 | 0.807 |

For comparison, in Python (sentence-transformers, float32), dense then hybrid:

| Model | scifact | scifact-fr | nfcorpus | squad-fr | hybrid scifact | hybrid nfcorpus | hybrid squad-fr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3 (568M) | 0.642 | 0.598 | 0.319 | 0.820 | 0.691 | 0.342 | 0.825 |
| Qwen3-Embedding-0.6B | 0.700 | 0.667 | 0.364 | 0.774 | 0.718 | 0.353 | 0.808 |
| granite-97m-multilingual-r2 | 0.685 | 0.588 | 0.297 | 0.727 | 0.715 | 0.332 | 0.797 |

granite-311m is clearly better across languages (French questions over English documents, +6 points), and not better within one language. On these sets, bekko a25m is the better compromise.

The indecis numbers match the Python reference (sentence-transformers in float32): 0.7598 for granite-311m on squad-fr in float32 on both sides, 0.7594 in int8; 0.6995 against 0.6977 for bekko a8m on scifact.

## Fine-tuning for search

A few hundred real (query, relevant document) pairs from the domain are worth more than a larger model. Trained on SciFact's 807 training claims (2 epochs, one BM25 hard negative per claim), then measured on its 300 test claims:

| Model | scifact | scifact-fr | nfcorpus | squad-fr | training |
| --- | --- | --- | --- | --- | --- |
| bekko a8m | 0.698 → 0.728 | 0.595 → 0.624 | 0.334 → 0.335 | 0.791 → 0.792 | 10 min |
| bekko a25m | 0.715 → **0.764** | 0.591 → **0.683** | 0.347 → 0.355 | 0.799 → 0.796 | 32 min |

The fine-tuned a25m beats every model tried, Qwen3-Embedding-0.6B included, on scifact and on French questions over the same corpus, although training used English claims only, and does not lose elsewhere. Training and test claims come from the same collection: this is the gain of in-domain data, not general progress.

Without labeled queries, an LLM can write them from the corpus itself: three queries per passage (a question, keywords, a claim, in the passage's language), plus one BM25 negative per query. With `cadoles/opencode-deepseek-v4.1-flash`, 10 passages per call, about 2 s per passage:

| bekko a8m trained on | scifact | scifact-fr | nfcorpus | squad-fr |
| --- | --- | --- | --- | --- |
| nothing | 0.698 | 0.595 | 0.334 | 0.791 |
| 5,442 synthetic queries on 1,494 SciFact passages (18 min) | 0.711 | 0.622 | 0.336 | 0.791 |
| 807 real SciFact training claims (10 min) | 0.728 | 0.624 | 0.335 | 0.792 |
| 2,733 synthetic queries on squad-fr's 761 passages (6 min) | 0.700 | 0.608 | 0.330 | 0.795 |

Trained on the 13,746 synthetic queries of all three corpora (1 epoch, 1 h 55), a25m goes from 0.715 / 0.591 / 0.347 / 0.799 to 0.727 / 0.665 / 0.346 / **0.813**: squad-fr passes granite-311m and nears bge-m3 (0.820), scifact-fr nears Qwen3-Embedding-0.6B (0.667).

Synthetic queries bring about half the gain of real ones within the language and as much across languages, and cost nothing elsewhere. The passages are those of the evaluation corpora (never their test queries): this measures adapting to a known corpus, the case of a search engine indexing its own documents.

```bash
bin/indecis train -search -backbone bekko-embedding-v1-a25m -train pairs.jsonl -out model -epochs 2
```

Each line of `pairs.jsonl` is a query and a relevant document, with optional ids and hard negatives:

```json
{"query_id": "q12", "query": "…", "document_id": "d7", "document": "…", "negatives": ["…"]}
```

In each batch, a query must prefer its document over the other queries' documents and the hard negatives (`SearchBatches`, then `FitEmbeddings`). Pairs of one query never share a batch, and a document relevant to a query is never counted as its negative. Keep the evaluation queries out of the training pairs.

## Speed

One P-core of a Core Ultra 7 265U, int8, SIMD build, compact models (`indecis compact -int8-embeddings`), while another process kept the other cores busy:

| Model | 13–14-token query | passages of up to 512 tokens | ready, in memory |
| --- | --- | --- | --- |
| bekko a8m | 1.7 ms | about 9,800 tokens/s | 17 MB |
| bekko a25m | 5.5 ms | about 3,300 tokens/s | 33 MB |
| granite-311m | 21.7 ms | about 890 tokens/s | 117 MB |

Indexing 20,000 passages of 300 tokens takes about 10 minutes per core with a8m, 30 with a25m and 1.9 hours with granite-311m; requests of several texts spread over the cores.

## From Go

```go
m, err := indecis.Open(dir, indecis.WithInt8(), indecis.WithMaxLen(512))
docs, err := m.EmbedDocuments(ctx, passages...)
qs, err := m.EmbedQueries(ctx, "Quel est le délai de préavis ?")
short := indecis.Truncate(qs[0], 256) // Matryoshka models only
```

`Embed` embeds the texts as given, without prompt. `EmbedQueries` and `EmbedDocuments` put the model's prompt first, as sentence-transformers does with `prompt_name`.

## Command line

```bash
bin/indecis embed -model granite -max-len 512 < passages.jsonl > vectors.jsonl
bin/indecis embed -model granite -query -dim 256 < questions.txt
bin/indecis embed -model granite -format f32 -in passages.jsonl -out vectors.f32
```

Input lines are plain text or JSON `{"id", "text"}`; the output is one JSON line `{"id", "embedding"}` per text, or the vectors end to end in float32 little-endian with `-format f32`.

## HTTP

`indecis-serve` answers OpenAI's embeddings API, so any OpenAI client (genai, the Python SDK, Amoxtli's `AMOXTLI_EVAL_EMBED_BASE_URL`) works by changing the URL:

```bash
bin/indecis-serve -model granite=granite-compact -max-len 512
curl -s localhost:8080/v1/embeddings -d '{"model": "granite", "input": ["Bonjour", "Hello"], "dimensions": 256}'
```

`input` is a string or an array of up to 2,048 strings; `encoding_format` is `float` or `base64`; `dimensions` truncates (Matryoshka). `"input_type": "query"`, an extension, puts the model's query prompt first. `usage.prompt_tokens` counts the tokens read.

## Memory

A raw backbone is decoded to float32 at load time: about 2 GB for granite-311m. `indecis compact` turns it into an indecis model whose weights are memory-mapped; bf16 weights stay bf16, without loss, and `-int8-embeddings` stores the 262,152 × 768 table in int8 (594 MB → 404 MB on disk):

```bash
bin/indecis compact -model granite-embedding-311m-multilingual-r2 -out granite-compact -int8-embeddings
```

Measured with `indecis-serve`: the compact granite-311m holds 117 MB once loaded (RssAnon), and peaked at 853 MB while embedding requests of 64 passages of up to 512 tokens. The int8 table changed squad-fr's nDCG@10 by 0.0004 (0.7594 → 0.7590).
