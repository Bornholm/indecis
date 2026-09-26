package teacher

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/bornholm/genai/llm"
)

// Command fait d'un harnais de code en ligne de commande (Claude Code, Pi…)
// un client de chat pour le teacher, comme les agents de Conclave : la
// commande reçoit le prompt, écrit sa réponse, et le client l'extrait.
//
// Les textes soumis au teacher sont des injections par construction. Un
// harnais a des outils (shell, fichiers) : la commande doit les désactiver
// entièrement, et elle s'exécute dans un répertoire temporaire vide. Le
// format pi-json permet en plus de vérifier qu'aucun outil n'a été appelé.
type Command struct {
	// Args est la commande complète, sans le prompt.
	Args []string
	// SystemFlag est l'option qui reçoit le prompt système
	// (« --system-prompt »). Vide : le prompt système est placé en tête du
	// message.
	SystemFlag string
	// Input est « stdin » (défaut), « argument » (le message en dernier
	// argument) ou « file » : le message est écrit dans prompt.md, dans le
	// répertoire temporaire, et passé comme « @prompt.md » (convention de
	// Pi). Un argument est limité à 128 Kio sous Linux : un lot de longs
	// textes ne tient pas en « argument ».
	Input string
	// Output est « claude-json » (enveloppe de claude --output-format json),
	// « pi-json » (événements de pi --mode json) ou « text ».
	Output string
	// Env complète l'environnement du processus.
	Env []string
	// Timeout borne chaque appel ; 5 minutes par défaut.
	Timeout time.Duration
}

// ErrToolUse signale qu'un harnais a appelé un outil malgré leur
// désactivation : sa réponse est rejetée.
var ErrToolUse = errors.New("teacher : le harnais a appelé un outil alors qu'ils sont désactivés")

// ChatCompletion implémente llm.ChatCompletionClient. Le format de réponse
// demandé n'est pas transmis : les prompts du teacher décrivent le JSON
// attendu et la réponse est extraite de la prose si besoin.
func (c *Command) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)
	var system, user []string
	for _, m := range opts.Messages {
		switch m.Role() {
		case llm.RoleSystem:
			system = append(system, m.Content())
		default:
			user = append(user, m.Content())
		}
	}
	args := append([]string(nil), c.Args...)
	message := strings.Join(user, "\n\n")
	if len(system) > 0 {
		if c.SystemFlag != "" {
			args = append(args, c.SystemFlag, strings.Join(system, "\n\n"))
		} else {
			message = strings.Join(system, "\n\n") + "\n\n" + message
		}
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("teacher : commande vide")
	}

	timeout := c.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	dir, err := os.MkdirTemp("", "indecis-teacher-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	switch c.Input {
	case "argument":
		args = append(args, "--", message)
	case "file":
		if err := os.WriteFile(dir+"/prompt.md", []byte(message), 0o600); err != nil {
			return nil, err
		}
		args = append(args, "@prompt.md")
	}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), c.Env...)
	if c.Input != "argument" && c.Input != "file" {
		cmd.Stdin = strings.NewReader(message)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		detail := lastLine(stderr.String())
		if detail == "" {
			detail = lastLine(stdout.String())
		}
		return nil, fmt.Errorf("teacher : %s : %w (%s)", args[0], err, detail)
	}

	text, err := extractText(c.Output, stdout.Bytes())
	if err != nil {
		return nil, err
	}
	return llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, text), nil), nil
}

func extractText(format string, out []byte) (string, error) {
	switch format {
	case "claude-json":
		var env struct {
			Type    string `json:"type"`
			Result  string `json:"result"`
			IsError bool   `json:"is_error"`
		}
		if err := json.Unmarshal(bytes.TrimSpace(out), &env); err != nil {
			return "", fmt.Errorf("teacher : enveloppe claude illisible : %w", err)
		}
		if env.IsError {
			return "", fmt.Errorf("teacher : claude a signalé une erreur : %s", lastLine(env.Result))
		}
		return env.Result, nil
	case "pi-json":
		var last string
		for _, line := range bytes.Split(out, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if len(line) == 0 || line[0] != '{' {
				continue
			}
			var ev struct {
				Type    string `json:"type"`
				Message *struct {
					Role    string `json:"role"`
					Content []struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"content"`
				} `json:"message"`
			}
			if json.Unmarshal(line, &ev) != nil {
				continue
			}
			switch ev.Type {
			case "tool_execution_start":
				return "", ErrToolUse
			case "message_end":
				if ev.Message == nil || ev.Message.Role != "assistant" {
					continue
				}
				var text string
				for _, c := range ev.Message.Content {
					if c.Type == "text" {
						text += c.Text
					}
				}
				if text != "" {
					last = text
				}
			}
		}
		if last == "" {
			return "", fmt.Errorf("teacher : pi n'a produit aucun message")
		}
		return last, nil
	case "", "text":
		return string(out), nil
	}
	return "", fmt.Errorf("teacher : format de sortie %q inconnu", format)
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	}
	if len(s) > 300 {
		s = s[:300]
	}
	return s
}

var _ llm.ChatCompletionClient = (*Command)(nil)
