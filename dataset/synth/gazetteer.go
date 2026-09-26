// Gazetteers : listes pondérées de valeurs, reprises de go-anon.
//
// Le format est un TSV « valeur <TAB> poids <TAB> metadata(JSON) », choisi pour
// rester lisible et diffable. Les poids bruts issus de
// sources statistiques sont très piqués : une poignée de valeurs écrase la
// longue traîne. L'exposant d'aplatissement rend ce compromis réglable.
package synth

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// Entry est une valeur du gazetteer avec son poids et ses métadonnées.
type Entry struct {
	Value    string
	Weight   float64
	Metadata map[string]string
}

// Gazetteer est une liste pondérée prête pour le tirage.
type Gazetteer struct {
	entries []Entry
	cum     []float64 // poids cumulés, pour la recherche binaire
	total   float64
}

// Options pilote la mise en forme de la distribution au chargement.
type GazetteerOptions struct {
	// Alpha aplatit les poids : w' = w^Alpha. 1 conserve la distribution
	// réelle, 0 la rend uniforme. Défaut 0.6.
	Alpha float64
	// MinWeight écarte les valeurs sous ce poids brut, pour couper le bruit
	// orthographique des fichiers sources.
	MinWeight float64
}

// DefaultOptions retourne les réglages recommandés.
func DefaultGazetteerOptions() GazetteerOptions { return GazetteerOptions{Alpha: 0.6} }

// NewGazetteer construit un Gazetteer à partir d'entrées déjà chargées.
func NewGazetteer(entries []Entry, opts GazetteerOptions) (*Gazetteer, error) {
	if opts.Alpha <= 0 {
		opts.Alpha = 1
	}
	s := &Gazetteer{}
	for _, e := range entries {
		if e.Weight < opts.MinWeight || e.Weight <= 0 {
			continue
		}
		w := math.Pow(e.Weight, opts.Alpha)
		s.total += w
		s.entries = append(s.entries, e)
		s.cum = append(s.cum, s.total)
	}
	if len(s.entries) == 0 {
		return nil, fmt.Errorf("gazetteer vide après filtrage (MinWeight=%v)", opts.MinWeight)
	}
	return s, nil
}

// LoadGazetteer lit un gazetteer au format TSV. Les lignes vides et celles commençant
// par « # » sont ignorées. Un poids absent vaut 1.
func LoadGazetteer(r io.Reader, opts GazetteerOptions) (*Gazetteer, error) {
	var entries []Entry
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimRight(sc.Text(), "\r")
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		cols := strings.Split(raw, "\t")
		e := Entry{Value: strings.TrimSpace(cols[0]), Weight: 1}
		if e.Value == "" {
			continue
		}
		if len(cols) > 1 && strings.TrimSpace(cols[1]) != "" {
			w, err := strconv.ParseFloat(strings.TrimSpace(cols[1]), 64)
			if err != nil {
				return nil, fmt.Errorf("ligne %d : poids invalide %q : %w", line, cols[1], err)
			}
			e.Weight = w
		}
		if len(cols) > 2 && strings.TrimSpace(cols[2]) != "" {
			if err := json.Unmarshal([]byte(cols[2]), &e.Metadata); err != nil {
				return nil, fmt.Errorf("ligne %d : metadata invalide : %w", line, err)
			}
		}
		entries = append(entries, e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return NewGazetteer(entries, opts)
}

// Subset construit un Gazetteer restreint aux entrées satisfaisant pred, en
// conservant leurs poids déjà aplatis.
//
// Sert à contraindre la cohérence sémantique : on ne compose pas un
// « Laboratoire de Travaux Publics ». Retourne nil si aucune entrée ne passe,
// à l'appelant de retomber sur le Gazetteer complet.
func (s *Gazetteer) Subset(pred func(Entry) bool) *Gazetteer {
	out := &Gazetteer{}
	for i, e := range s.entries {
		if !pred(e) {
			continue
		}
		w := s.cum[i]
		if i > 0 {
			w -= s.cum[i-1]
		}
		out.total += w
		out.entries = append(out.entries, e)
		out.cum = append(out.cum, out.total)
	}
	if len(out.entries) == 0 {
		return nil
	}
	return out
}

// Pick tire une entrée selon la distribution pondérée.
func (s *Gazetteer) Pick(rng *rand.Rand) Entry {
	target := rng.Float64() * s.total
	i := sort.SearchFloat64s(s.cum, target)
	if i >= len(s.entries) {
		i = len(s.entries) - 1
	}
	return s.entries[i]
}

// PickValue tire une entrée et n'en retourne que la valeur.
func (s *Gazetteer) PickValue(rng *rand.Rand) string { return s.Pick(rng).Value }

// Len retourne le nombre d'entrées retenues.
func (s *Gazetteer) Len() int { return len(s.entries) }
