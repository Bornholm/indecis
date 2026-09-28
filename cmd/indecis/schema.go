package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/bornholm/indecis"
)

// schemaFile is the command's schema file: a list of questions, or a
// model's indecis.json. An option can be a name, or an object
// {"name", "description", "examples"}: description and examples serve
// open mode (train -open, predict and eval with -schema).
type schemaFile struct {
	schema     indecis.Schema
	candidates map[string][]indecis.Candidate // question -> described options
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
			return nil, fmt.Errorf("%s: expected list of questions or indecis.json: %w", path, err)
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
					return nil, fmt.Errorf("%s: unreadable option of %s: %w", path, q.Name, err)
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

// open translates the schema into open questions (see Model.DecideOpen).
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
