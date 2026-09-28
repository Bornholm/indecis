package main

import (
	"context"
	"log"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// Descriptions of a few training categories: the model must learn to
// read a category named alone the same way as a described category.
var described = map[string]string{
	"Incident":                        "Something that worked has stopped working or is failing right now.",
	"Request":                         "The sender asks for something new: information, access, a service or an action.",
	"Problem":                         "A recurring or underlying issue whose cause must be found.",
	"Change":                          "A planned modification to a system, a configuration or a service.",
	"Billing and Payments":            "Invoices, payments, charges, refunds and subscription fees.",
	"Returns and Exchanges":           "Returning, exchanging or replacing a purchased product.",
	"Sales and Pre-Sales":             "Questions before buying: prices, quotes, product capabilities.",
	"Human Resources":                 "Employment, recruitment, leave, payroll and staff matters.",
	"Service Outages and Maintenance": "A service is down, degraded or under planned maintenance.",
	"Technical Support":               "Help with a technical problem on a product or a system.",
	"Customer Service":                "General help for customers about their account or their orders.",
	"General Inquiry":                 "A general question that fits no specific department.",
}

func candidate(name string, rng *rand.Rand) indecis.Candidate {
	c := indecis.Candidate{Name: name}
	if d, ok := described[name]; ok && rng.Intn(2) == 0 {
		c.Description = d
	}
	return c
}

// pairs turns tickets into (category, email) examples for the match
// question, over three lists: queues, types, keywords.
//
// Each category serves as a negative as often as a positive: negatives
// are drawn according to the frequency of positives. Without this, the
// model learns which names are often "true" (frequent queues) instead
// of matching the category to the email, and rejects any unknown
// category.
func pairs(tickets []dataset.Example, rng *rand.Rand) []dataset.Example {
	type item struct{ text, pos string }
	lists := map[string][]item{}
	for _, e := range tickets {
		lists["queue"] = append(lists["queue"], item{e.Text, e.Meta["queue"]})
		if t := e.Meta["type"]; t != "" {
			lists["type"] = append(lists["type"], item{e.Text, t})
		}
		if tags := strings.Split(e.Meta["tags"], "|"); tags[0] != "" {
			lists["tag"] = append(lists["tag"], item{e.Text, tags[rng.Intn(min(3, len(tags)))]})
		}
	}
	var out []dataset.Example
	for _, name := range []string{"queue", "type", "tag"} {
		items := lists[name]
		// Drawing a negative: the positive category of another example
		// from the same list, so according to the frequency of positives.
		for _, it := range items {
			var negs []indecis.Candidate
			for tries := 0; len(negs) < 1 && tries < 20; tries++ {
				n := items[rng.Intn(len(items))].pos
				if n != it.pos && !labels[it.text][n] {
					negs = append(negs, candidate(n, rng))
				}
			}
			out = append(out, indecis.CandidatePairs(question, it.text, candidate(it.pos, rng), negs)...)
		}
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

func train(ctx context.Context, dir, backbone, out string, n, epochs int) error {
	tickets, _, _, err := load(dir)
	if err != nil {
		return err
	}
	var pool []dataset.Example
	queues, types, tags := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, e := range tickets {
		if heldOutQueue(e.Meta["queue"]) || testTicket(e) {
			continue
		}
		pool = append(pool, e)
		queues[e.Meta["queue"]] = true
		if t := e.Meta["type"]; t != "" {
			types[t] = true
		}
		for _, t := range strings.Split(e.Meta["tags"], "|") {
			if t != "" {
				tags[t] = true
			}
		}
	}
	rng := rand.New(rand.NewSource(1))
	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	calibTickets := pool[:200]
	pool = pool[200:min(len(pool), 200+n)]
	labels = map[string]map[string]bool{}
	for _, e := range append(append([]dataset.Example(nil), pool...), calibTickets...) {
		l := map[string]bool{e.Meta["queue"]: true, e.Meta["type"]: true}
		for _, t := range strings.Split(e.Meta["tags"], "|") {
			l[t] = true
		}
		labels[e.Text] = l
	}
	trainEx := pairs(pool, rng)
	calib := pairs(calibTickets, rng)
	log.Printf("%d tickets -> %d training pairs (%d queues, %d types, %d keywords), %d calibration",
		len(pool), len(trainEx), len(queues), len(types), len(tags), len(calib))

	m, err := indecis.New(backbone, indecis.Schema{indecis.NewNoul(question, "Le courriel relève-t-il de cette catégorie ?")}, 1,
		indecis.WithPairs(), indecis.WithMaxLen(256))
	if err != nil {
		return err
	}
	opts := indecis.DefaultTrainOptions()
	opts.Epochs = epochs
	opts.Progress = func(p indecis.Progress) {
		if p.Step%100 == 0 || p.Step == p.Steps {
			log.Printf("step %d/%d loss %.4f - %.0f tokens/s, %s", p.Step, p.Steps, p.Loss,
				float64(p.Tokens)/p.Elapsed.Seconds(), p.Elapsed.Round(time.Second))
		}
	}
	if err := m.Fit(ctx, trainEx, opts); err != nil {
		return err
	}
	temps, err := m.Calibrate(ctx, calib)
	if err != nil {
		return err
	}
	log.Printf("temperature: %v", temps)
	if err := m.Save(out); err != nil {
		return err
	}
	log.Printf("model written to %s", out)
	return evaluate(ctx, dir, out, backbone)
}

// evaluate compares the pair model and the backbone's embeddings.
func evaluate(ctx context.Context, dir, model, backbone string) error {
	tickets, imnim, enron, err := load(dir)
	if err != nil {
		return err
	}
	m, err := indecis.Load(model, indecis.WithInt8())
	if err != nil {
		return err
	}
	base, err := indecis.New(backbone, indecis.Schema{indecis.NewNoul(question, "")}, 1, indecis.WithInt8())
	if err != nil {
		return err
	}
	for _, s := range evalSets(tickets, imnim, enron) {
		if err := report(ctx, "embeddings", embedChooser(base, 0.05), s); err != nil {
			return err
		}
		if err := report(ctx, "pairs", pairChooser(m), s); err != nil {
			return err
		}
	}
	return nil
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// labels gives, for each ticket text, all its categories (queue,
// type, keywords): none of them may serve as its negative.
var labels map[string]map[string]bool
