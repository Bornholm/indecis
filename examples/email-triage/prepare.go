package main

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/bornholm/indecis/dataset"
)

// prepare collecte les trois sources de la preuve de concept :
//
//   - tickets.jsonl : Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0),
//     tickets de support en anglais et en allemand, avec leur file (queue),
//     leur type et des mots-clés (tags). Licence non commerciale : jeu de
//     preuve de concept, jamais versionné.
//   - imnim.jsonl : imnim/multiclass-email-classification (MIT), courriels
//     avec des catégories que le modèle ne voit jamais à l'entraînement.
//   - enron.jsonl : courriels réels d'Enron (corbt/enron-emails), sans
//     étiquettes, que les teachers classent selon la liste classique.
func prepare(ctx context.Context, dir string, ticketPages, enronPages int) error {
	var tickets []dataset.Example
	err := rows(ctx, "Tobi-Bueck/customer-support-tickets", "train", 61765, ticketPages, "poc", func(r map[string]any) {
		text := emailText(str(r, "subject"), str(r, "body"))
		queue, typ := str(r, "queue"), str(r, "type")
		if text == "" || queue == "" {
			return
		}
		var tags []string
		for i := 1; i <= 8; i++ {
			if t := str(r, fmt.Sprintf("tag_%d", i)); t != "" {
				tags = append(tags, t)
			}
		}
		tickets = append(tickets, dataset.Example{Text: text, Meta: map[string]string{
			"source": "tickets", "lang": str(r, "language"), "queue": queue, "type": typ, "tags": strings.Join(tags, "|"),
		}})
	})
	if err != nil {
		return err
	}
	if err := write(dir, "tickets", dedupe(tickets)); err != nil {
		return err
	}

	var imnim []dataset.Example
	err = rows(ctx, "imnim/multiclass-email-classification", "train", 2105, 0, "", func(r map[string]any) {
		var labels []string
		if ls, ok := r["labels"].([]any); ok {
			for _, l := range ls {
				if s, ok := l.(string); ok {
					labels = append(labels, s)
				}
			}
		}
		text := emailText(str(r, "subject"), str(r, "body"))
		if text == "" || len(labels) == 0 {
			return
		}
		imnim = append(imnim, dataset.Example{Text: text, Meta: map[string]string{"source": "imnim", "labels": strings.Join(labels, "|")}})
	})
	if err != nil {
		return err
	}
	if err := write(dir, "imnim", dedupe(imnim)); err != nil {
		return err
	}

	var enron []dataset.Example
	err = rows(ctx, "corbt/enron-emails", "train", 517401, enronPages, "poc", func(r map[string]any) {
		body := quoted.Split(str(r, "body"), 2)[0]
		body = strings.TrimSpace(body)
		if len(body) < 80 || len(body) > 3000 {
			return
		}
		enron = append(enron, dataset.Example{Text: emailText(str(r, "subject"), body), Meta: map[string]string{"source": "enron"}})
	})
	if err != nil {
		return err
	}
	return write(dir, "enron", dedupe(enron))
}

// quoted repère le début d'un message cité ou transféré dans une réponse.
var quoted = regexp.MustCompile(`(?m)^\s*-{3,}\s*(Original Message|Forwarded by)`)

func emailText(subject, body string) string {
	body = strings.ReplaceAll(body, `\n`, "\n") // tickets : sauts de ligne échappés
	subject, body = strings.TrimSpace(subject), strings.TrimSpace(body)
	switch {
	case body == "":
		return ""
	case subject == "":
		return body
	}
	return subject + "\n\n" + body
}

func str(r map[string]any, k string) string {
	s, _ := r[k].(string)
	return strings.TrimSpace(s)
}

func write(dir, name string, ex []dataset.Example) error {
	path := filepath.Join(dir, name+".jsonl")
	if err := dataset.WriteFile(path, ex); err != nil {
		return err
	}
	log.Printf("%s : %d exemples → %s", name, len(ex), path)
	return nil
}

func dedupe(ex []dataset.Example) []dataset.Example {
	seen := map[string]bool{}
	out := ex[:0]
	for _, e := range ex {
		if !seen[e.Text] {
			seen[e.Text] = true
			out = append(out, e)
		}
	}
	return out
}

// moreEnron collecte d'autres courriels Enron pour l'entraînement, sans
// recouvrement avec ceux déjà collectés (dont la référence).
func moreEnron(ctx context.Context, dir string, pages int, seed string) error {
	known := map[string]bool{}
	if old, err := dataset.ReadFile(filepath.Join(dir, "enron.jsonl")); err == nil {
		for _, e := range old {
			known[e.Text] = true
		}
	}
	var out []dataset.Example
	err := rows(ctx, "corbt/enron-emails", "train", 517401, pages, seed, func(r map[string]any) {
		body := strings.TrimSpace(quoted.Split(str(r, "body"), 2)[0])
		if len(body) < 80 || len(body) > 3000 {
			return
		}
		text := emailText(str(r, "subject"), body)
		if !known[text] {
			known[text] = true
			out = append(out, dataset.Example{Text: text, Meta: map[string]string{"source": "enron"}})
		}
	})
	if err != nil {
		return err
	}
	return write(dir, "enron_train_unlabeled", out)
}
