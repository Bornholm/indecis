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

// Command turns a command-line coding harness (Claude Code, Pi...) into a
// chat client for the teacher, like Conclave's agents: the command receives
// the prompt, writes its answer, and the client extracts it.
//
// Texts submitted to the teacher are injections by construction. A harness
// has tools (shell, files): the command must disable them entirely, and it
// runs in an empty temporary directory. The pi-json format additionally
// allows checking that no tool was called.
type Command struct {
	// Args is the full command, without the prompt.
	Args []string
	// SystemFlag is the option that receives the system prompt
	// ("--system-prompt"). Empty: the system prompt is prepended to the
	// message.
	SystemFlag string
	// Input is "stdin" (default), "argument" (the message as the last
	// argument) or "file": the message is written to prompt.md, in the
	// temporary directory, and passed as "@prompt.md" (Pi's convention). An
	// argument is limited to 128 KiB on Linux: a batch of long texts does
	// not fit in "argument".
	Input string
	// Output is "claude-json" (claude --output-format json envelope),
	// "pi-json" (pi --mode json events) or "text".
	Output string
	// Env extends the process environment.
	Env []string
	// Timeout bounds each call; 5 minutes by default.
	Timeout time.Duration
}

// ErrToolUse signals that a harness called a tool despite them being
// disabled: its response is rejected.
var ErrToolUse = errors.New("teacher: the harness called a tool although tools are disabled")

// ChatCompletion implements llm.ChatCompletionClient. The requested response
// format is not passed through: the teacher's prompts describe the expected
// JSON and the response is extracted from prose if needed.
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
		return nil, fmt.Errorf("teacher: empty command")
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
		return nil, fmt.Errorf("teacher: %s: %w (%s)", args[0], err, detail)
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
			return "", fmt.Errorf("teacher: unreadable claude envelope: %w", err)
		}
		if env.IsError {
			return "", fmt.Errorf("teacher: claude reported an error: %s", lastLine(env.Result))
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
			return "", fmt.Errorf("teacher: pi produced no message")
		}
		return last, nil
	case "", "text":
		return string(out), nil
	}
	return "", fmt.Errorf("teacher: unknown output format %q", format)
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
