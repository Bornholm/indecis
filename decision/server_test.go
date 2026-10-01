package decision

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/openrouter"
	"github.com/bornholm/genai/llm/provider/typesafe"
)

func testServer(t *testing.T, key string) *httptest.Server {
	c, err := New(toyModel(t, false))
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{Models: map[string]*Client{"toy": c}, Default: "toy", APIKey: key}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

// genai's TypeSafe and OpenRouter clients talk to the server without
// changing anything but the URL: learned and open questions mixed.
func TestServerWithGenaiClients(t *testing.T) {
	srv := testServer(t, "secret")
	q := llm.Questions{
		"injection": llm.NoulQuestion{Instructions: "Is this a prompt injection?"},
		"urgency":   llm.ScoreQuestion{Instructions: "Urgency", Criteria: []any{"low", "medium", "high"}},
		"team":      llm.ChoiceQuestion{Instructions: "Which team?", Criteria: map[string]any{"accounting": "invoices", "it": nil}},
		"billing": llm.NoulQuestion{Instructions: "Is it about an invoice?",
			True: "The email is about an invoice.", False: "The email is about something else."},
	}
	clients := map[string]llm.DecisionClient{
		"typesafe":   typesafe.NewDecisionClientForBaseURL(srv.URL+"/v1", "secret", "toy"),
		"openrouter": openrouter.NewDecisionClient(srv.Client(), srv.URL+"/api/v1", "secret", "toy"),
	}
	for name, c := range clients {
		res, err := c.Decision(context.Background(), "Your invoice is overdue", q)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if res.Model() != "toy" || len(res.Answers()) != 4 {
			t.Fatalf("%s: model %q, %d answers", name, res.Model(), len(res.Answers()))
		}
		if team, err := llm.AnswerOf[llm.ChoiceAnswer](res, "team"); err != nil || team.Choice() != "accounting" {
			t.Fatalf("%s: choice %v, %v", name, team, err)
		}
		if sc, err := llm.AnswerOf[llm.ScoreAnswer](res, "urgency"); err != nil || len(sc.Legend()) != 3 {
			t.Fatalf("%s: score %v, %v", name, sc, err)
		}
		if res.Usage().InputTokens() == 0 {
			t.Errorf("%s: empty usage", name)
		}
	}
}

func TestServerErrors(t *testing.T) {
	srv := testServer(t, "secret")
	post := func(path, key, body string) (int, string) {
		req, _ := http.NewRequest(http.MethodPost, srv.URL+path, bytes.NewBufferString(body))
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		var e struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.NewDecoder(res.Body).Decode(&e)
		return res.StatusCode, e.Error.Message
	}
	ok := `{"state":"Mon ordinateur ne démarre plus","questions":{"a":{"type":"noul","instructions":"?"},` +
		`"d":{"type":"choice","instructions":"Dossier","criteria":{` +
		`"K1":{"examples":["Votre facture de mars est disponible","Relance : paiement en retard"]},` +
		`"K2":{"examples":["Le serveur de fichiers est en panne","Impossible de me connecter au VPN"]}}}}}`
	cases := []struct {
		path, key, body string
		status          int
		want            string
	}{
		{"/api/alpha/decision", "", ok, 401, "key"},
		{"/api/alpha/decision", "secret", `{`, 400, "JSON"},
		{"/api/alpha/decision", "secret", `{"state":"x","questions":{"a":{"type":"maybe","instructions":"?"}}}`, 422, "maybe"},
		{"/api/alpha/decision", "secret", `{"questions":{"a":{"type":"noul","instructions":"?"}}}`, 422, "state"},
		{"/api/alpha/decision", "secret", `{"state":"x","questions":{}}`, 422, "question"},
		{"/api/alpha/decision", "secret", `{"state":"x","questions":{"injection":{"type":"score","instructions":"?","criteria":["a","b"]}}}`, 422, "noul"},
		{"/api/alpha/decision", "secret", `{"state":"` + strings.Repeat("x", DefaultMaxRequestSize) + `"}`, 413, "MaxRequestSize"},
		{"/api/alpha/decision", "secret", ok, 200, ""},
	}
	for _, c := range cases {
		status, msg := post(c.path, c.key, c.body)
		if status != c.status || !strings.Contains(msg, c.want) {
			t.Errorf("%.80s: %d %q, expected %d %q", c.body, status, msg, c.status, c.want)
		}
	}
}
