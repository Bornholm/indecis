package decision

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"testing"

	"github.com/bornholm/genai/llm"
	genaiopenai "github.com/bornholm/genai/llm/provider/openai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// genai's OpenAI client, the one Amoxtli uses, reads /v1/embeddings
// without changing anything but the URL.
func TestEmbeddingsWithGenaiClient(t *testing.T) {
	srv := testServer(t, "secret")
	oc := openai.NewClient(option.WithBaseURL(srv.URL+"/v1/"), option.WithAPIKey("secret"))
	c := genaiopenai.NewEmbeddingsClient(oc, "toy")
	res, err := c.Embeddings(context.Background(), []string{"Your invoice is overdue", "Bonjour"})
	if err != nil {
		t.Fatal(err)
	}
	vecs := res.Embeddings()
	if len(vecs) != 2 || res.Usage().PromptTokens() == 0 {
		t.Fatalf("%d embeddings, usage %d", len(vecs), res.Usage().PromptTokens())
	}
	var n float64
	for _, x := range vecs[0] {
		n += x * x
	}
	if math.Abs(n-1) > 1e-5 {
		t.Fatalf("norm² %g", n)
	}
	short, err := c.Embeddings(context.Background(), []string{"Your invoice is overdue"}, llm.WithDimensions(8))
	if err != nil {
		t.Fatal(err)
	}
	if got := short.Embeddings()[0]; len(got) != 8 {
		t.Fatalf("%d dimensions, expected 8", len(got))
	}
}

func TestEmbeddingsFormats(t *testing.T) {
	srv := testServer(t, "")
	post := func(body string) (int, map[string]any) {
		res, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewBufferString(body))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		var out map[string]any
		json.NewDecoder(res.Body).Decode(&out)
		return res.StatusCode, out
	}
	code, f := post(`{"model":"toy","input":"hello"}`)
	if code != 200 {
		t.Fatalf("float: %d %v", code, f)
	}
	code, b := post(`{"model":"toy","input":["hello"],"encoding_format":"base64","input_type":"query"}`)
	if code != 200 {
		t.Fatalf("base64: %d %v", code, b)
	}
	floats := f["data"].([]any)[0].(map[string]any)["embedding"].([]any)
	enc := b["data"].([]any)[0].(map[string]any)["embedding"].(string)
	vec := make([]float32, len(floats))
	for i, x := range floats {
		vec[i] = float32(x.(float64))
	}
	if enc != encodeFloats(vec) {
		t.Fatal("base64 differs from the float encoding")
	}
	for _, c := range []struct {
		body string
		code int
	}{
		{`{"model":"toy","input":[1,2,3]}`, 422},
		{`{"model":"toy","input":[]}`, 422},
		{`{"model":"toy","input":"x","encoding_format":"int8"}`, 422},
		{`{"model":"toy","input":"x","dimensions":100000}`, 422},
		{`{"input":"x"}`, 200},
		{`not json`, 400},
	} {
		if code, out := post(c.body); code != c.code {
			t.Errorf("%s: %d, expected %d (%v)", c.body, code, c.code, out)
		}
	}
	res, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var list struct {
		Data []struct{ ID string } `json:"data"`
	}
	json.NewDecoder(res.Body).Decode(&list)
	if len(list.Data) != 1 || list.Data[0].ID != "toy" {
		t.Fatalf("models %v", list)
	}
}

func TestEmbedInputsRejectsMissingInput(t *testing.T) {
	for _, raw := range []string{``, `null`, ` null `, `[]`, `["a", null]`} {
		if _, err := embedInputs(json.RawMessage(raw)); err == nil {
			t.Errorf("input %q accepted", raw)
		}
	}
	if got, err := embedInputs(json.RawMessage(`""`)); err != nil || len(got) != 1 {
		t.Errorf(`input "": %v, %v`, got, err)
	}
}
