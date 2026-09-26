package teacher

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"sync"
)

// Cache garde les réponses du LLM dans un fichier JSONL, en ajout seul. Une
// génération relancée ne repaie que les appels nouveaux, et le fichier
// documente exactement ce que le teacher a répondu.
type Cache struct {
	mu   sync.Mutex
	path string
	data map[string]string
}

type cacheLine struct {
	Key      string `json:"key"`
	Response string `json:"response"`
}

// OpenCache ouvre, ou crée au premier Put, un cache.
func OpenCache(path string) (*Cache, error) {
	c := &Cache{path: path, data: map[string]string{}}
	f, err := os.Open(path)
	if errors.Is(err, fs.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for n := 1; sc.Scan(); n++ {
		var l cacheLine
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
			return nil, fmt.Errorf("teacher : cache %s, ligne %d : %w", path, n, err)
		}
		c.data[l.Key] = l.Response
	}
	return c, sc.Err()
}

// Get retourne une réponse mémorisée.
func (c *Cache) Get(key string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.data[key]
	return v, ok
}

// Put mémorise une réponse et l'ajoute au fichier.
func (c *Cache) Put(key, response string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.data[key]; ok {
		return nil
	}
	c.data[key] = response
	if c.path == "" {
		return nil
	}
	f, err := os.OpenFile(c.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	b, err := json.Marshal(cacheLine{Key: key, Response: response})
	if err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Len retourne le nombre de réponses mémorisées.
func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.data)
}

func cacheKey(model, system, user, schema string, temperature float64) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00%s\x00%s\x00%g", model, system, user, schema, temperature)
	return hex.EncodeToString(h.Sum(nil))
}
