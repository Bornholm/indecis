package decision

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/bornholm/indecis"
)

// MaxEmbedInputs bounds the texts of one /v1/embeddings request, as
// OpenAI does.
const MaxEmbedInputs = 2048

// embedRequest is OpenAI's embeddings request. input_type is an extension
// (the name Voyage and Cohere use): "query" puts the model's query prompt
// before each text, anything else is a document.
type embedRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
	Dimensions     int             `json:"dimensions"`
	InputType      string          `json:"input_type"`
}

type embedDatum struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"` // []float32, or base64 of the float32 little-endian
}

// embed serves POST /v1/embeddings, compatible with OpenAI's API.
func (s *Server) embed(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, "missing or invalid API key")
		return
	}
	limit := s.MaxRequestSize
	if limit <= 0 {
		limit = DefaultMaxRequestSize
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if int64(len(body)) > limit {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("request larger than %d bytes (Server.MaxRequestSize; indecis-serve -max-body)", limit))
		return
	}
	var req embedRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	texts, err := embedInputs(req.Input)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	if req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64" {
		writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("encoding_format %q: float or base64", req.EncodingFormat))
		return
	}
	name, c := s.client(req.Model)
	if c == nil {
		writeError(w, http.StatusNotFound, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	m, v, err := c.load(c.defaultDir)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if m == nil || v != nil {
		writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("model %q does not embed texts", name))
		return
	}
	if req.Dimensions < 0 || req.Dimensions > m.Dimension() {
		writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("dimensions %d: between 1 and %d", req.Dimensions, m.Dimension()))
		return
	}
	if s.slots != nil {
		select {
		case s.slots <- struct{}{}:
			defer func() { <-s.slots }()
		case <-r.Context().Done():
			writeError(w, http.StatusServiceUnavailable, "request abandoned while waiting for a slot")
			return
		}
	}
	var vecs [][]float32
	role := "document"
	if req.InputType == "query" {
		role = "query"
		vecs, err = m.EmbedQueries(r.Context(), texts...)
	} else {
		vecs, err = m.EmbedDocuments(r.Context(), texts...)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	// The tokens read: the prompt the model puts first included, truncation
	// applied.
	tokens := 0
	prompt := m.Info().Prompts[role]
	data := make([]embedDatum, len(vecs))
	for i, v := range vecs {
		n, err := m.Tokens(indecis.Input{Text: prompt + texts[i]})
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		tokens += n
		v = indecis.Truncate(v, req.Dimensions)
		data[i] = embedDatum{Object: "embedding", Index: i, Embedding: v}
		if req.EncodingFormat == "base64" {
			data[i].Embedding = encodeFloats(v)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  name,
		"usage":  map[string]int{"prompt_tokens": tokens, "total_tokens": tokens},
	})
	if s.Logger != nil {
		s.Logger.Info("embeddings", "model", name, "inputs", len(texts), "tokens", tokens, "duration", time.Since(start).Round(time.Microsecond))
	}
}

// embedInputs decodes "input": a string or an array of strings. Token
// arrays (OpenAI also accepts them) would be ids of another tokenizer.
func embedInputs(raw json.RawMessage) ([]string, error) {
	// A JSON null decodes into a string without error: it would be embedded
	// as an empty text.
	if t := bytes.TrimSpace(raw); len(t) == 0 || bytes.Equal(t, []byte("null")) {
		return nil, fmt.Errorf("input is missing")
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	// Pointers, so that a null element is told from an empty string.
	var elems []*string
	if err := json.Unmarshal(raw, &elems); err != nil {
		return nil, fmt.Errorf("input: a string or an array of strings (token arrays are not supported)")
	}
	many := make([]string, len(elems))
	for i, e := range elems {
		if e == nil {
			return nil, fmt.Errorf("input[%d] is null", i)
		}
		many[i] = *e
	}
	switch {
	case len(many) == 0:
		return nil, fmt.Errorf("input is empty")
	case len(many) > MaxEmbedInputs:
		return nil, fmt.Errorf("input: %d texts, at most %d per request", len(many), MaxEmbedInputs)
	}
	return many, nil
}

func encodeFloats(v []float32) string {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(b)
}

// openAIModels serves GET /v1/models in OpenAI's format.
func (s *Server) openAIModels(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, "missing or invalid API key")
		return
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	// Only the text models: /v1/embeddings refuses the others.
	list := []model{}
	for _, n := range s.names() {
		c := s.Models[n]
		m, v, err := c.load(c.defaultDir)
		if err != nil && s.Logger != nil {
			s.Logger.Warn("model left out of /v1/models: it does not load", "model", n, "error", err)
		}
		if err != nil || m == nil || v != nil {
			continue
		}
		list = append(list, model{ID: n, Object: "model", OwnedBy: "indecis"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": list})
}
