package main

import (
	"flag"
	"log"

	"github.com/bornholm/indecis/dataset"
)

// runSplit met de côté une part d'un jeu d'exemples, de façon
// reproductible : par famille (des familles entières, pour mesurer la
// généralisation à des familles jamais vues) ou exemple par exemple.
func runSplit(args []string) error {
	fs := flag.NewFlagSet("split", flag.ExitOnError)
	in := fs.String("in", "", "exemples (JSONL, motifs séparés par des virgules)")
	fraction := fs.Float64("fraction", 0.1, "part mise de côté")
	by := fs.String("by", "example", "example ou family")
	seed := fs.Uint64("seed", 1, "graine")
	kept := fs.String("kept", "", "fichier des exemples gardés")
	held := fs.String("held", "", "fichier des exemples mis de côté")
	fs.Parse(args)
	if err := required(fs, "in", "kept", "held"); err != nil {
		return err
	}
	ex, err := readExamples(*in)
	if err != nil {
		return err
	}
	k, h := holdOut(ex, *fraction, *seed, *by)
	if *by != "family" {
		// holdOut a vidé les familles pour découper : on les rend.
		k, h = restoreFamilies(ex, k), restoreFamilies(ex, h)
	}
	if err := dataset.WriteFile(*kept, k); err != nil {
		return err
	}
	if err := dataset.WriteFile(*held, h); err != nil {
		return err
	}
	log.Printf("%d gardés → %s, %d mis de côté → %s", len(k), *kept, len(h), *held)
	return nil
}

func restoreFamilies(orig, part []dataset.Example) []dataset.Example {
	fam := map[string]string{}
	for _, e := range orig {
		fam[e.Context+"\x00"+e.Text] = e.Family
	}
	for i := range part {
		part[i].Family = fam[part[i].Context+"\x00"+part[i].Text]
	}
	return part
}
