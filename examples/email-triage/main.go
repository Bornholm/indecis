// Command email-triage est une preuve de concept : classer des courriels
// parmi des catégories données à l'inférence, qui peuvent changer d'un appel
// à l'autre (voir indecis.ChooseAmong).
//
//	go run ./examples/email-triage prepare
//	go run ./examples/email-triage baseline
//	go run ./examples/email-triage train
//	go run ./examples/email-triage eval -model …
package main

import (
	"context"
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
		fmt.Fprintln(os.Stderr, "usage : email-triage prepare|baseline|train|eval [options]")
		os.Exit(2)
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	home, _ := os.UserHomeDir()
	data := fs.String("data", filepath.Join(home, ".cache/indecis/datasets/email"), "répertoire des données")
	backbone := fs.String("backbone", filepath.Join(home, ".cache/indecis/models/bekko-embedding-v1-a8m"), "backbone")
	ticketPages := fs.Int("ticket-pages", 80, "prepare : pages de 100 tickets")
	enronPages := fs.Int("enron-pages", 20, "prepare : pages de 100 courriels Enron")
	temperature := fs.Float64("temperature", 0.05, "baseline : température du softmax sur les cosinus")
	out := fs.String("out", filepath.Join(home, ".cache/indecis/runs/email-pairs"), "train, eval : répertoire du modèle")
	n := fs.Int("n", 3000, "train : tickets d'entraînement")
	epochs := fs.Int("epochs", 1, "train : époques")
	shots := fs.Int("k", 5, "few-shot : exemples par catégorie")
	fs.Parse(os.Args[2:])
	ctx := context.Background()
	if err := os.MkdirAll(*data, 0o755); err != nil {
		log.Fatal(err)
	}
	var err error
	switch cmd {
	case "prepare":
		err = prepare(ctx, *data, *ticketPages, *enronPages)
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
		err = fmt.Errorf("commande %q inconnue", cmd)
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

// baseline évalue la comparaison de plongements avec le backbone tel quel.
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
		if err := report(ctx, "plongements", embedChooser(m, temperature), s); err != nil {
			return err
		}
	}
	return nil
}
