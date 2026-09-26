// Package messages extrait les textes à analyser d'une requête de chat au
// format OpenAI, en distinguant leur provenance.
//
// Le découpage suit celui de prompt-guard : dernier tour utilisateur, tours
// utilisateur précédents, résultats d'outils. Deux détecteurs chaînés doivent
// juger les mêmes textes, sinon leur fusion compare des choses différentes.
package messages

import (
	"encoding/json"
	"strings"
)

type message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// Turns regroupe les textes d'une requête par provenance.
type Turns struct {
	// System est le prompt système (messages system et developer, joints) :
	// le périmètre fixé par l'opérateur.
	System string
	// Last est le dernier message utilisateur.
	Last string
	// Earlier sont les messages utilisateur précédents, du plus ancien au
	// plus récent.
	Earlier []string
	// Tools sont les résultats d'outils, dans l'ordre : le texte que le
	// modèle lit sans que personne dans la conversation l'ait écrit.
	Tools []string
}

// Parse lit messages_json. Une requête illisible donne des Turns vides : le
// détecteur n'a alors rien à dire, ce n'est pas à lui de la refuser.
func Parse(messagesJSON string) Turns {
	var msgs []message
	if err := json.Unmarshal([]byte(messagesJSON), &msgs); err != nil {
		return Turns{}
	}
	var t Turns
	var users []string
	for _, m := range msgs {
		switch m.Role {
		case "user":
			if s := textOf(m.Content); s != "" {
				users = append(users, s)
			}
		case "system", "developer":
			if s := textOf(m.Content); s != "" {
				if t.System != "" {
					t.System += "\n"
				}
				t.System += s
			}
		case "tool":
			if s := textOf(m.Content); s != "" {
				t.Tools = append(t.Tools, s)
			}
		}
	}
	if n := len(users); n > 0 {
		t.Last = users[n-1]
		t.Earlier = users[:n-1]
	}
	return t
}

// textOf accepte un contenu en chaîne ou en tableau de parties typées, et ne
// garde que les parties texte.
func textOf(content json.RawMessage) string {
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return strings.TrimSpace(s)
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type != "text" || p.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Text)
	}
	return strings.TrimSpace(b.String())
}
