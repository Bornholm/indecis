# Producing data

An indecis model learns from examples in JSONL format: a text, an optional context, and one label per question.

```json
{"text": "My parcel has not arrived.", "labels": {"topic": "delivery", "human": false}}
{"context": "You are the assistant of an online shop.", "text": "Tell me a joke.", "labels": {"off_role": true}, "family": "off_role/humor"}
```

`family` groups related examples (from the same template, the same assistant). `indecis split -by family` then sets whole families aside, to measure how well the model generalizes to families it never saw. `meta` holds free-form information, such as the source, which `indecis eval -by source` can break the numbers down by.

Two tools produce examples: templates, and LLMs that label real texts.

## Templates

The `dataset/synth` package (command `indecis synth`) produces examples labeled by construction. Its engine comes from go-anon. A template declares its labels in its header, and each branch can change them:

```
family: message/delivery
lang: en
label.topic: delivery
---
{{one:Hello, |}}{{one}}my order has not arrived.{{label:urgency=medium}}{{|}}this is the third time the parcel gets lost!{{label:urgency=high}}{{/one}}
```

| Directive | Effect |
| --- | --- |
| `{{one:a\|b\|c}}` | one alternative, as plain text; `\n` is a line break |
| `{{one}}…{{\|}}…{{/one}}` | one alternative that may contain other directives |
| `{{pick:city}}`, `{{pick:city:x}}` | a value from the `city.tsv` gazetteer; a named slot keeps the same value across the example |
| `{{int:1-100}}`, `{{digits:6}}` | numbers |
| `[?name:0.3]…[/]` | a section present with probability 0.3 |
| `{{x:base64}}…{{/x}}` | transforms the rendered text: `base64`, `leet`, `homoglyph`, `zwsp`, `rot13`, `hex`, `spaced`, `upper`, `lower`, `noise` |
| `{{include:family/*\|p=0.4}}` | renders another template and adds its labels |
| `{{label:name=value}}` | sets a label from the rendered branch |
| `{{user}}` | what comes before becomes the context of the example |

When a template includes another one, a true boolean label wins: a harmless email that includes an attack becomes an attack. A mistake (missing gazetteer, include with no target, stray `}}`) makes loading fail instead of producing wrong examples.

Templates have a limit: a model quickly learns their phrasings. In our measurements, a template corpus scores 100% on its own held-out part, which says nothing about real texts. Always test on texts the templates did not produce.

## Labeling by LLMs (teachers)

The `teacher` module, kept separate so the library has no dependencies, has LLMs label texts. Its command is `indecis-teach`:

```bash
cd teacher && go build -o ../bin/indecis-teach ./cmd/indecis-teach && cd ..

# one LLM, configured through GENAI_* variables (genai library)
bin/indecis-teach label -in texts.jsonl -out labeled.jsonl -schema schema.json

# two command-line coding agents, and their consensus
bin/indecis-teach label -teachers teacher/teachers.example.yaml -guidelines policy.md \
    -schema schema.json -in texts.jsonl -out consensus.jsonl -disagreements to-review.jsonl
```

With `-teachers`, each teacher labels the texts without seeing existing labels. The output keeps only what they agree on. Their disagreements, and the cases where they all contradict an existing label, go to `-disagreements` with each opinion. Human review can focus on these cases, about 10% of the texts in our runs.

Teachers can be coding agents such as Claude Code or Pi, as in [Conclave](https://github.com/bornholm/conclave), each with its own model and subscription. `teacher/teachers.example.yaml` configures two of them. They work in batches of 20 texts, at a throttled pace (`interval`, `concurrency`) to spare quotas. Answers are cached: running a command again costs nothing, and `-cache-only` replays the cache without any call.

`indecis-teach rewrite` has texts rewritten, for instance translated, while keeping their labels.

### What we learned

- **Write the labeling policy down and give it to the teachers** (`-guidelines`). Without it, each teacher applies its own idea of the question. On a corpus of prompts, two teachers contradicted the source on 27% of the texts, often wrongly. With the policy, 5%, and their consensus agreed with a human reviewer 97% of the time.
- **Keep exact labels.** When a label comes from a template, a teacher that contradicts it is usually wrong: out of 11 contradictions, the teacher was wrong 9 times.
- **Paraphrasing templates does not add diversity**, and even lowered the scores on real texts: a teacher that rephrases sometimes softens the text, which keeps its label anyway. Varied real texts, labeled by consensus, help more.
- **Translating labeled real texts** helps little. In our email experiments, a few points at best.

### Security

Texts to label may contain instructions aimed at the model that reads them, and a coding agent can act on the machine. The example configuration turns every tool off (`--tools ""` for Claude, `--no-tools` for Pi), with no MCP server and no context file. Each call runs in an empty temporary directory, and a Pi answer that shows a tool call is rejected.
