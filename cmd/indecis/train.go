package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func runTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	backbone := fs.String("backbone", "", "backbone in transformers format (config.json, model.safetensors, tokenizer.json)")
	schemaPath := fs.String("schema", "", "schema: JSON list of questions (see the guide)")
	trainGlobs := fs.String("train", "", "training examples (JSONL, comma-separated patterns)")
	calibGlobs := fs.String("calib", "", "calibration examples (default: 5% of training)")
	testGlobs := fs.String("test", "", "test examples (default: 10% of training, by family)")
	holdout := fs.Float64("holdout", 0.1, "fraction held out for test without -test")
	holdoutBy := fs.String("holdout-by", "example", "set aside by example, or by family (whole families: measures generalization to families never seen)")
	out := fs.String("out", "", "directory of the trained model")
	open := fs.Bool("open", false, "train embeddings for options given at inference (ChooseNearest, DecideOpen) rather than fixed heads")
	pairs := fs.Bool("pairs", false, "read (context, text) pairs")
	maxLen := fs.Int("max-len", 256, "maximum tokens read per example")
	def := indecis.DefaultTrainOptions()
	epochs := fs.Int("epochs", def.Epochs, "epochs")
	batch := fs.Int("batch", def.BatchSize, "examples per batch")
	lr := fs.Float64("lr", def.LR, "encoder learning rate (default in -open mode: 2e-5)")
	headLR := fs.Float64("head-lr", def.HeadLR, "learning rate of the heads")
	seed := fs.Int64("seed", def.Seed, "seed")
	threads := fs.Int("threads", 0, "cores (0: all)")
	int8Emb := fs.Bool("int8-embeddings", false, "write the embeddings table in int8 (twice as small)")
	fs.Parse(args)
	if err := required(fs, "backbone", "schema", "train", "out"); err != nil {
		return err
	}
	ctx := context.Background()
	sf, err := readSchema(*schemaPath)
	if err != nil {
		return err
	}
	all, err := readExamples(*trainGlobs)
	if err != nil {
		return err
	}
	trainEx, test := all, []dataset.Example(nil)
	if *testGlobs != "" {
		if test, err = readExamples(*testGlobs); err != nil {
			return err
		}
	} else if *holdout > 0 {
		trainEx, test = holdOut(all, *holdout, uint64(*seed), *holdoutBy)
	}
	var calib []dataset.Example
	if *calibGlobs != "" {
		if calib, err = readExamples(*calibGlobs); err != nil {
			return err
		}
	} else if !*open {
		trainEx, calib = holdOut(trainEx, 0.05, uint64(*seed)+1, *holdoutBy)
	}
	log.Printf("%d training examples, %d calibration, %d test", len(trainEx), len(calib), len(test))

	opts := []indecis.Option{indecis.WithMaxLen(*maxLen), indecis.WithThreads(*threads)}
	if *pairs {
		opts = append(opts, indecis.WithPairs())
	}
	m, err := indecis.New(*backbone, sf.schema, *seed, opts...)
	if err != nil {
		return err
	}
	to := indecis.DefaultTrainOptions()
	to.Epochs, to.BatchSize, to.HeadLR, to.Seed = *epochs, *batch, *headLR, *seed
	to.LR = *lr
	lrSet := false
	fs.Visit(func(f *flag.Flag) { lrSet = lrSet || f.Name == "lr" })
	if *open && !lrSet {
		to.LR = 2e-5 // the backbone embeddings are already good: we fine-tune them
	}
	start := time.Now()
	to.Progress = func(p indecis.Progress) {
		if p.Step%50 == 0 || p.Step == p.Steps {
			log.Printf("step %d/%d  loss %.4f  %.0f tokens/s  %s", p.Step, p.Steps, p.Loss,
				float64(p.Tokens)/p.Elapsed.Seconds(), p.Elapsed.Round(time.Second))
		}
	}
	if *open {
		batches := indecis.ChoiceBatches(trainEx, sf.open(), *batch, *seed)
		if len(batches) == 0 {
			return fmt.Errorf("no labeled example for the schema questions")
		}
		log.Printf("open mode: %d batches", len(batches))
		if err := m.FitEmbeddings(ctx, batches, to); err != nil {
			return err
		}
	} else {
		if err := m.Fit(ctx, trainEx, to); err != nil {
			return err
		}
		if len(calib) > 0 {
			temps, err := m.Calibrate(ctx, calib)
			if err != nil {
				return err
			}
			log.Printf("temperatures: %v", temps)
		}
	}
	log.Printf("training: %s", time.Since(start).Round(time.Second))
	if *int8Emb {
		indecis.WithInt8Embeddings()(m)
	}
	if err := m.Save(*out); err != nil {
		return err
	}
	log.Printf("model written to %s", *out)
	if len(test) > 0 {
		return report(ctx, m, sf, test, *open)
	}
	return nil
}

// report prints the measurements of a model on labeled examples: by
// its heads, or in open mode by the schema options.
func report(ctx context.Context, m *indecis.Model, sf *schemaFile, ex []dataset.Example, open bool) error {
	if !open {
		metrics, err := m.Evaluate(ctx, ex)
		if err != nil {
			return err
		}
		for _, mt := range metrics {
			fmt.Println(mt)
		}
		return nil
	}
	for _, q := range sf.open() {
		var texts []string
		var gold []int
		for _, e := range ex {
			if v, ok := e.Labels[q.Name]; ok {
				if j := q.OptionIndex(v); j >= 0 {
					texts = append(texts, e.Text)
					gold = append(gold, j)
				}
			}
		}
		if len(texts) == 0 {
			continue
		}
		d, err := m.DecideOpen(ctx, []indecis.OpenQuestion{q}, texts...)
		if err != nil {
			return err
		}
		correct := 0
		for i, dec := range d {
			a := dec[q.Name]
			var got string
			switch q.Kind {
			case indecis.Noul:
				got = map[bool]string{true: q.Options[0].Name, false: q.Options[1].Name}[a.P >= 0.5]
			default:
				got = a.Choice
			}
			if got == q.Options[gold[i]].Name {
				correct++
			}
		}
		fmt.Printf("%-14s n=%-5d options=%-3d acc=%.1f%%\n", q.Name, len(texts), len(q.Options), 100*float64(correct)/float64(len(texts)))
	}
	return nil
}

// holdOut sets aside a fraction of the examples, by family or example
// by example (Family cleared: HoldOut then splits on the text).
func holdOut(ex []dataset.Example, fraction float64, seed uint64, by string) (kept, held []dataset.Example) {
	if by != "family" {
		keyed := make([]dataset.Example, len(ex))
		for i, e := range ex {
			e.Family = ""
			keyed[i] = e
		}
		ex = keyed
	}
	return dataset.HoldOut(ex, fraction, seed)
}
