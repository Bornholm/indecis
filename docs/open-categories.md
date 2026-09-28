# Open categories

A classic `choice` question has fixed options, learned during training. Open mode classifies among a list given at each call: you can add a category, rename it, or describe it differently without retraining.

## How it works

The model turns the text into a vector, and each option too: its name, its description, and its examples if it has any. The closest option wins. The option vectors are computed once, so the cost does not depend on how many options there are.

```go
cats := []indecis.Candidate{
    {Name: "Billing", Description: "Invoices, payments, reminders",
        Examples: []string{"Reminder: unpaid invoice", "Your invoice #4521"}},
    {Name: "IT", Examples: []string{"The printer is broken"}},
}
set, _ := m.PrepareCandidates(ctx, cats)   // once
a, _ := m.ChooseIn(ctx, set, "The VPN is not responding")
a[0].Choice // "IT"
```

`ChooseNearest` does both steps in one call. `DecideOpen` asks full questions (yes/no, choice, score) described the same way. From the command line: `indecis predict -schema`, and `indecis eval -schema` to measure.

The backbone alone can already match a text with a description. `FitEmbeddings` (`indecis train -open`) fine-tunes it for the task: in each batch, a text must prefer its right option over every other option of the batch. The model learns to read a list, not one particular list, so train it on varied lists.

## What examples are worth

We measured it on three email datasets, with category lists the model had never seen (proof of concept `examples/email-triage`):

| Test set | a8m, name only | a8m, name and 5 examples | a25m, name only | a25m, name and 5 examples |
| --- | --- | --- | --- | --- |
| Enron emails, 13 categories in French | 43.3% | 47.8% | 52.8% | 58.0% |
| imnim, 10 categories | 77.4% | 93.8% | 76.8% | 96.8% |
| support tickets, 52 queues | 46.8% | 72.0% | 50.6% | 77.0% |

Without fine-tuning, a8m gets 27.9% on Enron and 69.2% on imnim with names only. A few examples per category help more than anything else: try them first.

The fine-tuned models learned from support tickets and from 2,000 Enron emails labeled by LLMs along eight varied lists, in French and English. The Enron test emails, and their list of 13 categories, were never used in training.

## Limits

- **The probabilities are not calibrated.** They come from a softmax over similarities with a fixed scale. To catch a text that fits no option, set a threshold on `Answer.Score` (the similarity of the chosen option) or on `Answer.Margin`, tuned on a few dozen examples.
- **Yes/no questions work poorly**: embeddings barely tell a statement from its negation. Prefer a choice between described options.
- **Neighboring categories** (billing and refunds) get mixed up more than distinct ones. Examples help, and so does a25m, at three times the compute.

## Why not a model that reads the (option, text) pair

`ChooseAmong` has a pair model judge each (option, text) pair. In our tests, it learned which option names were often right during training, and rejected unknown names: under 5% on imnim. It is also 10 to 50 times slower, since it makes one pass per option. For open options, embeddings win.
