// Command indecis crée, évalue et utilise de petits modèles de décision.
//
//	indecis synth    -templates dir -n 10000 -out data.jsonl
//	indecis train    -backbone dir -schema schema.json -train 'data/*.jsonl' -out model
//	indecis eval     -model model -data test.jsonl
//	indecis predict  -model model < textes.jsonl
//	indecis compact  -model model -out model-compact -int8-embeddings
//	indecis split    -in data.jsonl -fraction 0.2 -by family -kept train.jsonl -held test.jsonl
//
// Le guide docs/guide-creer-un-modele.md enchaîne ces étapes sur un
// exemple. L'étiquetage par des LLM (teachers) est fait par indecis-teach,
// module teacher, et le service HTTP par indecis-serve, module decision.
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
	{"synth", "génère des exemples étiquetés à partir de gabarits", runSynth},
	{"train", "entraîne un modèle (schéma fixe, ou plongements avec -open)", runTrain},
	{"eval", "mesure un modèle sur des exemples étiquetés", runEval},
	{"predict", "répond aux questions pour des textes (JSONL ou lignes)", runPredict},
	{"compact", "réduit la place d'un modèle (vocabulaire, table int8)", runCompact},
	{"split", "met de côté une part d'un jeu d'exemples (par famille ou par exemple)", runSplit},
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
	fmt.Fprintf(os.Stderr, "commande %q inconnue\n\n", os.Args[1])
	usage()
	os.Exit(2)
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage : indecis <commande> [options]\n\nCommandes :")
	for _, c := range commands {
		fmt.Fprintf(os.Stderr, "  %-8s %s\n", c.name, c.summary)
	}
	fmt.Fprintln(os.Stderr, "\nindecis <commande> -h détaille les options d'une commande.")
}

// readExamples lit des fichiers JSONL désignés par des motifs séparés par
// des virgules (~ est développé).
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
			return nil, fmt.Errorf("aucun fichier pour %q", g)
		}
		slices.Sort(files)
		for _, f := range files {
			ex, err := dataset.ReadFile(f)
			if err != nil {
				return nil, fmt.Errorf("%s : %w", f, err)
			}
			out = append(out, ex...)
		}
	}
	return out, nil
}

func required(fs *flag.FlagSet, names ...string) error {
	for _, n := range names {
		if fs.Lookup(n).Value.String() == "" {
			return fmt.Errorf("-%s est obligatoire (indecis %s -h)", n, fs.Name())
		}
	}
	return nil
}
