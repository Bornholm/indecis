package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"time"
)

// rows lit des pages de 100 lignes d'un jeu Hugging Face (datasets-server).
// pages = 0 : tout le jeu ; sinon des pages tirées au hasard, selon seed.
func rows(ctx context.Context, name, split string, total, pages int, seed string, fn func(map[string]any)) error {
	n := (total + 99) / 100
	offsets := make([]int, 0, n)
	if pages == 0 || pages >= n {
		for p := 0; p < n; p++ {
			offsets = append(offsets, p*100)
		}
	} else {
		h := fnv.New64a()
		fmt.Fprint(h, name, seed)
		for _, p := range rand.New(rand.NewSource(int64(h.Sum64()))).Perm(n)[:pages] {
			offsets = append(offsets, p*100)
		}
	}
	for _, off := range offsets {
		u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=%s&offset=%d&length=100",
			url.QueryEscape(name), split, off)
		var page struct {
			Rows []struct {
				Row map[string]any `json:"row"`
			} `json:"rows"`
		}
		if err := getJSON(ctx, u, &page); err != nil {
			return err
		}
		for _, r := range page.Rows {
			fn(r.Row)
		}
		time.Sleep(time.Second) // ménage le datasets-server
	}
	return nil
}

func getJSON(ctx context.Context, u string, v any) error {
	delay := 5 * time.Second
	for attempt := 1; ; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		if res.StatusCode == http.StatusOK {
			err = json.NewDecoder(res.Body).Decode(v)
			res.Body.Close()
			return err
		}
		res.Body.Close()
		if (res.StatusCode != http.StatusTooManyRequests && res.StatusCode < 500) || attempt == 6 {
			return fmt.Errorf("%s : HTTP %d", u, res.StatusCode)
		}
		log.Printf("HTTP %d, nouvel essai dans %s", res.StatusCode, delay)
		time.Sleep(delay)
		delay *= 2
	}
}
