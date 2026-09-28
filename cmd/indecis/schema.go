package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/bornholm/indecis"
)

// schemaFile est le fichier de schéma de la commande : une liste de
// questions, ou le indecis.json d'un modèle. Une option peut être un nom, ou
// un objet {"name", "description", "examples"} : description et exemples
// servent au mode ouvert (train -open, predict et eval avec -schema).
type schemaFile struct {
	schema     indecis.Schema
	candidates map[string][]indecis.Candidate // question → options décrites
}

type fileQuestion struct {
	Name         string            `json:"name"`
	Kind         indecis.Kind      `json:"kind"`
	Instructions string            `json:"instructions"`
	Options      []json.RawMessage `json:"options"`
}

func readSchema(path string) (*schemaFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var list []fileQuestion
	if err := json.Unmarshal(b, &list); err != nil {
		var meta struct {
			Schema []fileQuestion `json:"schema"`
		}
		if err2 := json.Unmarshal(b, &meta); err2 != nil || len(meta.Schema) == 0 {
			return nil, fmt.Errorf("%s : liste de questions ou indecis.json attendu : %w", path, err)
		}
		list = meta.Schema
	}
	sf := &schemaFile{candidates: map[string][]indecis.Candidate{}}
	for _, q := range list {
		var names []string
		var cands []indecis.Candidate
		for _, raw := range q.Options {
			var c indecis.Candidate
			if json.Unmarshal(raw, &c.Name) != nil {
				if err := json.Unmarshal(raw, &c); err != nil {
					return nil, fmt.Errorf("%s : option de %s illisible : %w", path, q.Name, err)
				}
			}
			names = append(names, c.Name)
			cands = append(cands, c)
		}
		sf.schema = append(sf.schema, indecis.Question{Name: q.Name, Kind: q.Kind, Instructions: q.Instructions, Options: names})
		sf.candidates[q.Name] = cands
	}
	if err := sf.schema.Validate(); err != nil {
		return nil, fmt.Errorf("%s : %w", path, err)
	}
	return sf, nil
}

// open traduit le schéma en questions ouvertes (voir Model.DecideOpen).
func (sf *schemaFile) open() []indecis.OpenQuestion {
	var out []indecis.OpenQuestion
	for _, q := range sf.schema {
		oq := indecis.OpenQuestion{Name: q.Name, Kind: q.Kind, Instructions: q.Instructions, Options: sf.candidates[q.Name]}
		if q.Kind == indecis.Noul && len(oq.Options) != 2 {
			oq.Options = []indecis.Candidate{
				{Name: "true", Description: "Oui : " + q.Instructions},
				{Name: "false", Description: "Non : " + q.Instructions},
			}
		}
		out = append(out, oq)
	}
	return out
}
