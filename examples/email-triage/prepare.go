package main

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/bornholm/indecis/dataset"
)

// prepare collects the proof of concept's three sources:
//
//   - tickets.jsonl: Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0),
//     support tickets in English and German, with their queue, their
//     type and keywords (tags). Non-commercial license: proof of
//     concept dataset, never versioned.
//   - imnim.jsonl: imnim/multiclass-email-classification (MIT), emails
//     with categories the model never sees during training.
//   - enron.jsonl: real Enron emails (corbt/enron-emails), unlabeled,
//     that the teachers classify according to the classic list.
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

// quoted spots the start of a quoted or forwarded message in a reply.
var quoted = regexp.MustCompile(`(?m)^\s*-{3,}\s*(Original Message|Forwarded by)`)

func emailText(subject, body string) string {
	body = strings.ReplaceAll(body, `\n`, "\n") // tickets: escaped line breaks
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
	log.Printf("%s: %d examples -> %s", name, len(ex), path)
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

// moreEnron collects more Enron emails for training, without overlap
// with those already collected (including the reference set).
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

// prepareASN collects follow-up inspection letters from the French
// nuclear safety authority (AdrienB134/ASN_Lettres_De_Suivi): real
// professional letters in French, whose subject gives the inspection
// theme. The text kept is the inspection summary, without the
// sentences that name the theme; the theme is the label to recover.
func prepareASN(ctx context.Context, dir string, pages, themes int) error {
	var all []dataset.Example
	err := rows(ctx, "AdrienB134/ASN_Lettres_De_Suivi", "train", 14408, pages, "poc", func(r map[string]any) {
		raw := str(r, "raw_file")
		m := asnTheme.FindStringSubmatch(raw)
		if m == nil {
			return
		}
		theme := normTheme(m[1])
		i := strings.Index(raw, "Synth")
		if i < 0 {
			return
		}
		body := raw[i:]
		if j := strings.IndexByte(body, '\n'); j > 0 {
			body = body[j+1:] // section title
		}
		var kept []string
		for _, s := range sentence.Split(body, -1) {
			if !strings.Contains(strings.ToLower(s), "thème") && !strings.Contains(s, theme) {
				kept = append(kept, strings.TrimSpace(s))
			}
		}
		text := strings.Join(kept, ". ")
		if len(text) < 200 {
			return
		}
		if len(text) > 1500 {
			text = text[:1500]
		}
		all = append(all, dataset.Example{Text: text, Meta: map[string]string{"source": "asn", "theme": theme}})
	})
	if err != nil {
		return err
	}
	count := map[string]int{}
	for _, e := range all {
		count[e.Meta["theme"]]++
	}
	var names []string
	for t := range count {
		names = append(names, t)
	}
	sort.Slice(names, func(i, j int) bool { return count[names[i]] > count[names[j]] })
	keep := map[string]bool{}
	for _, t := range names[:min(themes, len(names))] {
		keep[t] = true
	}
	var out []dataset.Example
	for _, e := range all {
		if keep[e.Meta["theme"]] {
			out = append(out, e)
		}
	}
	return write(dir, "asn", dedupe(out))
}

var (
	asnTheme = regexp.MustCompile(`sur le th[èe]me\s*[«"]\s*([^»"]{3,80}?)\s*[»"]`)
	sentence = regexp.MustCompile(`\.\s+`)
)

// normTheme unifies the case and variants of the same theme.
func normTheme(t string) string {
	t = strings.ToLower(strings.Join(strings.Fields(t), " "))
	t = strings.TrimPrefix(t, "gestion des ")
	if t == "" {
		return t
	}
	r := []rune(t)
	return strings.ToUpper(string(r[0])) + string(r[1:])
}
