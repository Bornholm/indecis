package main

import (
	"flag"
	"log"
	"os"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/dataset/synth"
)

func runSynth(args []string) error {
	fs := flag.NewFlagSet("synth", flag.ExitOnError)
	dir := fs.String("templates", "", "directory of templates (*.tmpl) and gazetteers (*.tsv)")
	n := fs.Int("n", 1000, "number of examples")
	seed := fs.Uint64("seed", 1, "seed")
	out := fs.String("out", "-", "output JSONL file (- : standard output)")
	dedupe := fs.Bool("dedupe", true, "discard duplicates")
	fs.Parse(args)
	if err := required(fs, "templates"); err != nil {
		return err
	}
	c, err := synth.LoadFS(os.DirFS(*dir), synth.DefaultGazetteerOptions())
	if err != nil {
		return err
	}
	ex, err := c.Generate(*n, synth.Options{Seed: *seed, Dedupe: *dedupe})
	if err != nil {
		return err
	}
	if *out == "-" {
		return dataset.WriteJSONL(os.Stdout, ex)
	}
	if err := dataset.WriteFile(*out, ex); err != nil {
		return err
	}
	log.Printf("%d examples -> %s", len(ex), *out)
	return nil
}
