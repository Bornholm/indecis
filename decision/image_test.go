package decision

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/bornholm/indecis/vision"
)

// An image model behind the decision API: the state is a data URL, the
// questions are open.
func TestServerImage(t *testing.T) {
	dir := os.Getenv("INDECIS_SIGLIP2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256")
	}
	if !vision.IsModel(dir) {
		t.Skipf("SigLIP 2 model absent (%s): set INDECIS_SIGLIP2_DIR", dir)
	}
	v, err := vision.Load(dir, vision.WithInt8())
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{Models: map[string]*Client{"images": FromVision("images", v)}, Default: "images"}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)

	png, err := os.ReadFile("../testdata/siglip2/shapes.png")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]any{
		"model": "images",
		"state": map[string]any{"image": "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)},
		"questions": map[string]any{
			"content": map[string]any{"type": "choice", "instructions": "What does the image show?", "criteria": map[string]any{
				"shapes": "a red circle on a blue background",
				"cat":    "a photo of a cat",
				"noise":  "random colored noise",
			}},
			"circle": map[string]any{"type": "noul", "instructions": "a red circle"},
		},
	})
	res, err := http.Post(srv.URL+"/api/alpha/decisions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out struct {
		Answers map[string]struct {
			Choice *string  `json:"choice"`
			Noul   *float64 `json:"noul"`
		} `json:"answers"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil || res.StatusCode != 200 {
		t.Fatalf("status %d, %v", res.StatusCode, err)
	}
	if c := out.Answers["content"].Choice; c == nil || *c != "shapes" {
		t.Errorf("content = %v, want shapes", c)
	}
	if p := out.Answers["circle"].Noul; p == nil || *p < 0.5 {
		t.Errorf("circle = %v, want > 0.5", p)
	}

	// Text where an image is expected: refused with 422.
	body, _ = json.Marshal(map[string]any{"state": "just text", "questions": map[string]any{"circle": map[string]any{"type": "noul", "instructions": "a red circle"}}})
	res2, err := http.Post(srv.URL+"/api/alpha/decisions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	res2.Body.Close()
	if res2.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("text state: status %d, want 422", res2.StatusCode)
	}
}
