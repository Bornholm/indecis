package decision

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"time"

	"github.com/bornholm/genai/llm"
)

// Server expose des modèles indecis en HTTP, avec l'API de décision de
// TypeSafe telle qu'OpenRouter la relaie : un client de ces services
// (genai, le SDK TypeSafe, un simple curl) peut viser un modèle local en
// changeant seulement l'URL.
//
//	POST /api/alpha/decisions   (OpenRouter ; /api/alpha/decision accepté)
//	POST /v1/systemone          (TypeSafe)
//	GET  /api/alpha/models      (modèles servis)
//
// Le champ « model » de la requête choisit le modèle par son nom ; vide ou
// inconnu d'un serveur à un seul modèle, c'est le modèle par défaut.
type Server struct {
	// Models associe un nom à un client ; Default nomme celui qui répond
	// quand la requête n'en désigne aucun.
	Models  map[string]*Client
	Default string
	// APIKey, si non vide, est exigée en « Authorization: Bearer … ».
	APIKey string
	// MaxConcurrent borne les décisions calculées en même temps (0 : pas
	// de borne) ; les autres attendent leur tour. Chaque décision en cours
	// tient ses propres tampons (≈ 10 Mo pour 256 tokens) : la borne fixe
	// aussi la mémoire de pointe.
	MaxConcurrent int
	slots         chan struct{}
	Logger        *slog.Logger
}

// maxRequestSize borne le corps d'une requête.
const maxRequestSize = 4 << 20

// Handler retourne le routeur HTTP du serveur.
func (s *Server) Handler() http.Handler {
	if s.MaxConcurrent > 0 && s.slots == nil {
		s.slots = make(chan struct{}, s.MaxConcurrent)
	}
	mux := http.NewServeMux()
	for _, p := range []string{"/api/alpha/decisions", "/api/alpha/decision", "/v1/systemone"} {
		mux.HandleFunc("POST "+p, s.decide)
	}
	mux.HandleFunc("GET /api/alpha/models", s.models)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	return mux
}

type wireQuestion struct {
	Type         llm.QuestionType `json:"type"`
	Instructions any              `json:"instructions"`
	Criteria     json.RawMessage  `json:"criteria"`
}

type wireRequest struct {
	State     any                     `json:"state"`
	Model     string                  `json:"model"`
	Questions map[string]wireQuestion `json:"questions"`
}

type wireAnswer struct {
	Type          llm.QuestionType   `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence"`
}

type wireUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

type wireResponse struct {
	Model   string                `json:"model"`
	Answers map[string]wireAnswer `json:"answers"`
	Usage   wireUsage             `json:"usage"`
}

func (s *Server) decide(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, "clé d'API absente ou invalide")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestSize+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(body) > maxRequestSize {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("requête de plus de %d octets", maxRequestSize))
		return
	}
	var req wireRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "JSON invalide : "+err.Error())
		return
	}
	if req.State == nil {
		writeError(w, http.StatusUnprocessableEntity, "state est obligatoire")
		return
	}
	questions, err := toQuestions(req.Questions)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	if s.slots != nil {
		select {
		case s.slots <- struct{}{}:
			defer func() { <-s.slots }()
		case <-r.Context().Done():
			writeError(w, http.StatusServiceUnavailable, "requête abandonnée en attente d'un créneau")
			return
		}
	}
	name, client := s.client(req.Model)
	if client == nil {
		writeError(w, http.StatusNotFound, fmt.Sprintf("modèle %q inconnu (servis : %v)", req.Model, s.names()))
		return
	}
	res, err := client.Decision(r.Context(), req.State, questions)
	if err != nil {
		status := http.StatusInternalServerError
		var ve llm.ValidationError
		if errors.As(err, &ve) {
			status = http.StatusUnprocessableEntity
		}
		writeError(w, status, err.Error())
		return
	}
	out := wireResponse{Model: name, Answers: map[string]wireAnswer{}}
	for id, a := range res.Answers() {
		out.Answers[id] = toWire(a)
	}
	if u := res.Usage(); u != nil {
		out.Usage = wireUsage{InputTokens: u.InputTokens(), OutputTokens: u.OutputTokens(), TotalTokens: u.TotalTokens()}
	}
	writeJSON(w, http.StatusOK, out)
	if s.Logger != nil {
		s.Logger.Info("décision", "model", name, "questions", len(questions), "durée", time.Since(start).Round(time.Microsecond))
	}
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, "clé d'API absente ou invalide")
		return
	}
	type model struct {
		ID        string   `json:"id"`
		Default   bool     `json:"default,omitempty"`
		Questions []string `json:"learned_questions"`
	}
	var list []model
	for _, n := range s.names() {
		m, _ := s.Models[n].model(s.Models[n].defaultDir)
		var qs []string
		if m != nil {
			for _, q := range m.Schema() {
				qs = append(qs, fmt.Sprintf("%s (%s)", q.Name, q.Kind))
			}
		}
		list = append(list, model{ID: n, Default: n == s.Default, Questions: qs})
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": list})
}

func (s *Server) authorized(r *http.Request) bool {
	return s.APIKey == "" || r.Header.Get("Authorization") == "Bearer "+s.APIKey
}

func (s *Server) client(model string) (string, *Client) {
	if c, ok := s.Models[model]; ok {
		return model, c
	}
	if model == "" || len(s.Models) == 1 {
		return s.Default, s.Models[s.Default]
	}
	return "", nil
}

func (s *Server) names() []string {
	out := make([]string, 0, len(s.Models))
	for n := range s.Models {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// toQuestions décode les questions du format TypeSafe.
func toQuestions(in map[string]wireQuestion) (llm.Questions, error) {
	out := llm.Questions{}
	for id, q := range in {
		switch q.Type {
		case llm.QuestionTypeNoul:
			var c struct {
				True  any `json:"true"`
				False any `json:"false"`
			}
			if len(q.Criteria) > 0 && string(q.Criteria) != "null" {
				if err := json.Unmarshal(q.Criteria, &c); err != nil {
					return nil, fmt.Errorf("questions.%s.criteria : objet {true, false} attendu", id)
				}
			}
			out[id] = llm.NoulQuestion{Instructions: q.Instructions, True: c.True, False: c.False}
		case llm.QuestionTypeChoice:
			var c map[string]any
			if err := json.Unmarshal(q.Criteria, &c); err != nil {
				return nil, fmt.Errorf("questions.%s.criteria : objet {option: description} attendu", id)
			}
			out[id] = llm.ChoiceQuestion{Instructions: q.Instructions, Criteria: c}
		case llm.QuestionTypeScore:
			var c []any
			if err := json.Unmarshal(q.Criteria, &c); err != nil {
				return nil, fmt.Errorf("questions.%s.criteria : liste de niveaux attendue", id)
			}
			out[id] = llm.ScoreQuestion{Instructions: q.Instructions, Criteria: c}
		default:
			return nil, fmt.Errorf("questions.%s.type : %q inconnu (noul, choice ou score)", id, q.Type)
		}
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return out, nil
}

func toWire(a llm.Answer) wireAnswer {
	switch v := a.(type) {
	case llm.NoulAnswer:
		p := v.Noul()
		return wireAnswer{Type: llm.QuestionTypeNoul, Noul: &p, Confidence: max(p, 1-p)}
	case llm.ChoiceAnswer:
		c := v.Choice()
		return wireAnswer{Type: llm.QuestionTypeChoice, Choice: &c, Probabilities: v.Probabilities(), Confidence: v.Confidence()}
	case llm.ScoreAnswer:
		sc := v.Score()
		return wireAnswer{Type: llm.QuestionTypeScore, Score: &sc, Legend: v.Legend(), Probabilities: v.Probabilities(), Confidence: v.Confidence()}
	}
	return wireAnswer{}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": message, "code": status}})
}
