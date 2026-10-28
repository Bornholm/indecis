// Command indecis creates, evaluates and uses small decision models.
//
//	indecis synth    -templates dir -n 10000 -out data.jsonl
//	indecis train    -backbone dir -schema schema.json -train 'data/*.jsonl' -out model
//	indecis eval     -model model -data test.jsonl
//	indecis predict  -model model < texts.jsonl
//	indecis compact  -model model -out model-compact -int8-embeddings
//	indecis split    -in data.jsonl -fraction 0.2 -by family -kept train.jsonl -held test.jsonl
//	indecis train-vision -backbone siglip -schema schema.json -train images.jsonl -out model
//	indecis check
//
// The guide docs/creating-a-model.md goes through these steps on an example.
// Labeling by LLMs (teachers) is done by indecis-teach, in the teacher
// module, and the HTTP service by indecis-serve, in the decision module.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/bornholm/indecis/dataset"
)

var commands = []struct {
	name, summary string
	run           func(args []string) error
}{
	{"synth", "generates labeled examples from templates", runSynth},
	{"train", "trains a model (fixed schema, or embeddings with -open)", runTrain},
	{"eval", "measures a model on labeled examples", runEval},
	{"predict", "answers questions for texts (JSONL or lines)", runPredict},
	{"compact", "reduces the size of a model (vocabulary, int8 table)", runCompact},
	{"split", "sets aside a part of a set of examples (by family or by example)", runSplit},
	{"train-vision", "trains a head that answers questions on images (SigLIP encoder)", runTrainVision},
	{"check", "checks that this processor runs the SIMD and assembly kernels", runCheck},
	{"version", "prints the version", runVersion},
}

// version is set at build time (-ldflags "-X main.version=...").
var version = "dev"

func runVersion(args []string) error {
	fmt.Println(version)
	return nil
}

func main() {
	log.SetFlags(log.Ltime)
	if len(os.Args) < 2 || os.Args[1] == "-h" || os.Args[1] == "help" {
		usage()
		os.Exit(2)
	}
	for _, c := range commands {
		if c.name == os.Args[1] {
			if err := c.run(os.Args[2:]); err != nil {
				log.Fatal(err)
			}
			return
		}
	}
	fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
	usage()
	os.Exit(2)
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: indecis <command> [options]\n\nCommands:")
	for _, c := range commands {
		fmt.Fprintf(os.Stderr, "  %-8s %s\n", c.name, c.summary)
	}
	fmt.Fprintln(os.Stderr, "\nindecis <command> -h details the options of a command.")
}

// readExamples reads JSONL files designated by comma-separated
// patterns (~ is expanded).
func readExamples(globs string) ([]dataset.Example, error) {
	var out []dataset.Example
	for _, g := range strings.Split(globs, ",") {
		g = strings.TrimSpace(g)
		if g == "" {
			continue
		}
		if strings.HasPrefix(g, "~/") {
			home, _ := os.UserHomeDir()
			g = filepath.Join(home, g[2:])
		}
		files, err := filepath.Glob(g)
		if err != nil {
			return nil, err
		}
		if len(files) == 0 {
			return nil, fmt.Errorf("no file for %q", g)
		}
		slices.Sort(files)
		for _, f := range files {
			ex, err := dataset.ReadFile(f)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", f, err)
			}
			out = append(out, ex...)
		}
	}
	return out, nil
}

func required(fs *flag.FlagSet, names ...string) error {
	for _, n := range names {
		if fs.Lookup(n).Value.String() == "" {
			return fmt.Errorf("-%s is required (indecis %s -h)", n, fs.Name())
		}
	}
	return nil
}
