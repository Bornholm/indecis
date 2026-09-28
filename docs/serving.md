# Serving a model

## HTTP server (`indecis-serve`)

`indecis-serve` exposes one or more models with the TypeSafe decision API, the one OpenRouter relays. A client written for Jev works by changing only the URL.

```bash
make serve
bin/indecis-serve -model support=model -addr 127.0.0.1:8080

curl -s localhost:8080/api/alpha/decisions -d '{
  "model": "support",
  "state": "My parcel has not arrived",
  "questions": {"topic": {"type": "choice", "instructions": "Topic",
    "criteria": {"delivery": null, "billing": null}}}
}'
```

| Path | Use |
| --- | --- |
| `POST /api/alpha/decisions` | OpenRouter API (`/api/alpha/decision` also accepted) |
| `POST /v1/systemone` | TypeSafe API |
| `GET /api/alpha/models` | served models and their learned questions |
| `GET /healthz` | liveness check |

A question named like a learned question goes through the model's head, which is calibrated. Any other question is asked in open mode: its criteria are compared with the state (see [open-categories.md](open-categories.md)). A criterion can carry examples: `{"description": "…", "examples": ["…"]}`. The TypeSafe API already accepts an object as a description; the `examples` field is an indecis convention, which another provider reads as a plain description.

`-model` also accepts a raw backbone such as bekko, which then answers open questions only.

| Option | Default | Effect |
| --- | --- | --- |
| `-model name=dir` | | served model, repeatable; the first one is the default |
| `-addr` | `127.0.0.1:8080` | listen address |
| `-api-key` | none | requires `Authorization: Bearer <key>` (or `INDECIS_API_KEY`) |
| `-threads` | 0 (all) | maximum cores per request |
| `-int8` | true | int8 compute if the CPU has AVX-VNNI |
| `-max-len` | the model's | tokens read per text |
| `-max-concurrent` | number of cores | decisions in progress; the others wait |
| `-memory-limit` | none | soft memory limit, in MiB |
| `-embed-cache` | 4,096 | option embeddings kept in memory |
| `-batching` | false | groups simultaneous requests (see [inference.md](inference.md)) |

Step by step, with the email classification model: [tutorial-email-classification.md](tutorial-email-classification.md).

## Decision provider for genai

The `decision` module exposes an indecis model as an `llm.DecisionClient` of [genai](https://github.com/bornholm/genai) (branch `feat/decision-client`). The same code then queries Jev or a local model:

```go
import _ "github.com/bornholm/indecis/decision"
```

```bash
GENAI_DECISION_PROVIDER=indecis
GENAI_DECISION_INDECIS_MODEL=/path/to/model
```

The adapter follows the same rule as the server: learned questions go through the heads, other questions are asked in open mode. For a learned question, it checks the type, and for a `choice` that the requested options belong to the model's; the distribution is then renormalized over that subset.

A string state is judged as is. An object `{"context": …, "text": …}` gives a pair model a (system prompt, message) pair. Any other value is serialized to JSON. `llm.WithDecisionModel(dir)` picks another model for one call.

The genai `typesafe` and `openrouter` clients are tested against `indecis-serve`.
