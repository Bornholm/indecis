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
	dir := fs.String("templates", "", "répertoire des gabarits (*.tmpl) et gazetteers (*.tsv)")
	n := fs.Int("n", 1000, "nombre d'exemples")
	seed := fs.Uint64("seed", 1, "graine")
	out := fs.String("out", "-", "fichier JSONL de sortie (- : sortie standard)")
	dedupe := fs.Bool("dedupe", true, "écarter les doublons")
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
	log.Printf("%d exemples → %s", len(ex), *out)
	return nil
}
