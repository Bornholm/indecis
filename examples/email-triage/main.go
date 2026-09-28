// Command email-triage is a proof of concept: classifying emails among
// categories given at inference time, which can change from one call to
// the next (see indecis.ChooseAmong).
//
//	go run ./examples/email-triage prepare
//	go run ./examples/email-triage baseline
//	go run ./examples/email-triage train
//	go run ./examples/email-triage eval -model …
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: email-triage prepare|baseline|train|eval [options]")
		os.Exit(2)
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	home, _ := os.UserHomeDir()
	data := fs.String("data", filepath.Join(home, ".cache/indecis/datasets/email"), "data directory")
	backbone := fs.String("backbone", filepath.Join(home, ".cache/indecis/models/bekko-embedding-v1-a8m"), "backbone")
	ticketPages := fs.Int("ticket-pages", 80, "prepare: pages of 100 tickets")
	enronPages := fs.Int("enron-pages", 20, "prepare: pages of 100 Enron emails")
	temperature := fs.Float64("temperature", 0.05, "baseline: softmax temperature over cosines")
	out := fs.String("out", filepath.Join(home, ".cache/indecis/runs/email-pairs"), "train, eval: model directory")
	n := fs.Int("n", 3000, "train: training tickets")
	epochs := fs.Int("epochs", 1, "train: epochs")
	shots := fs.Int("k", 5, "few-shot: examples per category")
	fs.BoolVar(&mix.fr, "fr", false, "train-embed: add Enron emails translated to French")
	fs.IntVar(&mix.synth, "synth", 0, "train-embed: synthetic French emails to add")
	fs.StringVar(&mix.synthDir, "synth-dir", "examples/email-triage/synth", "train-embed: templates")
	fs.Parse(os.Args[2:])
	ctx := context.Background()
	dataDir = *data
	if err := os.MkdirAll(*data, 0o755); err != nil {
		log.Fatal(err)
	}
	var err error
	switch cmd {
	case "prepare":
		err = prepare(ctx, *data, *ticketPages, *enronPages)
	case "teacher-schema":
		err = writeTeacherSchema(*data)
	case "asn":
		err = prepareASN(ctx, *data, 100, 12)
	case "more-enron":
		err = moreEnron(ctx, *data, *enronPages, "train")
	case "train":
		err = train(ctx, *data, *backbone, *out, *n, *epochs)
	case "train-embed":
		err = trainEmbed(ctx, *data, *backbone, *out, *n, *epochs)
	case "eval-embed":
		err = evaluateEmbed(ctx, *data, *out, *backbone)
	case "few-shot":
		err = evaluateFewShot(ctx, *data, *out, *backbone, *shots)
	case "eval":
		err = evaluate(ctx, *data, *out, *backbone)
	case "baseline":
		err = baseline(ctx, *data, *backbone, *temperature)
	default:
		err = fmt.Errorf("unknown command %q", cmd)
	}
	if err != nil {
		log.Fatal(err)
	}
}

func load(dir string) (tickets, imnim, enron []dataset.Example, err error) {
	if tickets, err = dataset.ReadFile(filepath.Join(dir, "tickets.jsonl")); err != nil {
		return
	}
	if imnim, err = dataset.ReadFile(filepath.Join(dir, "imnim.jsonl")); err != nil {
		return
	}
	enron, err = dataset.ReadFile(filepath.Join(dir, "enron.jsonl"))
	if labeled, e := dataset.ReadFile(filepath.Join(dir, "enron_labeled.jsonl")); e == nil {
		enron = labeled
	}
	return
}

// baseline evaluates embedding comparison with the backbone as is.
func baseline(ctx context.Context, dir, backbone string, temperature float64) error {
	tickets, imnim, enron, err := load(dir)
	if err != nil {
		return err
	}
	m, err := indecis.New(backbone, indecis.Schema{indecis.NewNoul(question, "")}, 1, indecis.WithInt8())
	if err != nil {
		return err
	}
	for _, s := range evalSets(tickets, imnim, enron) {
		if err := report(ctx, "embeddings", embedChooser(m, temperature), s); err != nil {
			return err
		}
	}
	return nil
}

func writeTeacherSchema(dir string) error {
	s, g := teacherSchema()
	b, err := json.MarshalIndent(s, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "train_schema.json"), b, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "train_guidelines.md"), []byte(g), 0o644)
}
