package main

import (
	"flag"
	"log"

	"github.com/bornholm/indecis/dataset"
)

// runSplit sets aside a part of a set of examples, reproducibly: by
// family (whole families, to measure generalization to families never
// seen) or example by example.
func runSplit(args []string) error {
	fs := flag.NewFlagSet("split", flag.ExitOnError)
	in := fs.String("in", "", "examples (JSONL, comma-separated patterns)")
	fraction := fs.Float64("fraction", 0.1, "fraction set aside")
	by := fs.String("by", "example", "example or family")
	seed := fs.Uint64("seed", 1, "seed")
	kept := fs.String("kept", "", "file of kept examples")
	held := fs.String("held", "", "file of held-out examples")
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
		// holdOut cleared the families to split: we restore them.
		k, h = restoreFamilies(ex, k), restoreFamilies(ex, h)
	}
	if err := dataset.WriteFile(*kept, k); err != nil {
		return err
	}
	if err := dataset.WriteFile(*held, h); err != nil {
		return err
	}
	log.Printf("%d kept -> %s, %d held out -> %s", len(k), *kept, len(h), *held)
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
