// Command synthsample écrit un échantillon du corpus synthétique de
// l'exemple prompt-injection, au format JSONL : c'est le texte réel des
// fixtures de parité du tokenizer.
package main

import (
	"flag"
	"log"
	"os"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/dataset/synth"
)

func main() {
	dir := flag.String("synth", "examples/prompt-injection/synth", "gabarits")
	n := flag.Int("n", 600, "nombre d'exemples")
	seed := flag.Uint64("seed", 42, "graine")
	out := flag.String("out", "", "fichier JSONL de sortie")
	flag.Parse()
	c, err := synth.LoadFS(os.DirFS(*dir), synth.DefaultGazetteerOptions())
	if err != nil {
		log.Fatal(err)
	}
	ex, err := c.Generate(*n, synth.Options{Seed: *seed, Dedupe: true})
	if err != nil {
		log.Fatal(err)
	}
	if err := dataset.WriteFile(*out, ex); err != nil {
		log.Fatal(err)
	}
}
