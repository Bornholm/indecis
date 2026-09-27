package indecis

import (
	"container/list"
	"sync"
)

// lru est un cache des plongements par texte, borné aux n derniers textes
// distincts. Un cache nil ne garde rien.
type lru struct {
	mu    sync.Mutex
	n     int
	order *list.List // du plus récent au plus ancien
	items map[string]*list.Element
}

type lruEntry struct {
	key string
	v   []float32
}

func newLRU(n int) *lru {
	return &lru{n: n, order: list.New(), items: map[string]*list.Element{}}
}

func (c *lru) get(k string) ([]float32, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[k]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(e)
	return e.Value.(*lruEntry).v, true
}

func (c *lru) put(k string, v []float32) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.items[k]; ok {
		e.Value.(*lruEntry).v = v
		c.order.MoveToFront(e)
		return
	}
	c.items[k] = c.order.PushFront(&lruEntry{k, v})
	for c.order.Len() > c.n {
		last := c.order.Back()
		c.order.Remove(last)
		delete(c.items, last.Value.(*lruEntry).key)
	}
}

func (c *lru) clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.order.Init()
	clear(c.items)
}
