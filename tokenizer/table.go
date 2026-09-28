package tokenizer

import (
	"hash/maphash"
	"slices"
)

// The vocabulary holds 256,000 strings and as many merges. As
// map[string]int32 and map[uint64]merge, it occupied 34 MB of heap; the
// tables below occupy 8.

var tableSeed = maphash.MakeSeed()

// strTable maps strings to ids: the strings are concatenated into a
// single block, and an open-addressing table finds the id of a string.
type strTable struct {
	blob  string
	off   []uint32 // string of id i: blob[off[i]:off[i+1]]
	slots []int32  // id + 1, 0 if free
	mask  uint64
}

// newStrTable indexes byID (id -> string); empty strings are absent ids,
// which lookup never finds.
func newStrTable(byID []string) *strTable {
	t := &strTable{off: make([]uint32, len(byID)+1)}
	size := 0
	for _, s := range byID {
		size += len(s)
	}
	blob := make([]byte, 0, size)
	for i, s := range byID {
		blob = append(blob, s...)
		t.off[i+1] = uint32(len(blob))
	}
	t.blob = string(blob)
	n := 1
	for n < 2*len(byID) {
		n <<= 1
	}
	t.slots = make([]int32, n)
	t.mask = uint64(n - 1)
	for id, s := range byID {
		if s == "" {
			continue
		}
		for h := maphash.String(tableSeed, s) & t.mask; ; h = (h + 1) & t.mask {
			if t.slots[h] == 0 {
				t.slots[h] = int32(id) + 1
				break
			}
		}
	}
	return t
}

func (t *strTable) str(id int32) string {
	return t.blob[t.off[id]:t.off[id+1]]
}

func (t *strTable) lookup(s string) (int32, bool) {
	if s == "" {
		return 0, false
	}
	for h := maphash.String(tableSeed, s) & t.mask; ; h = (h + 1) & t.mask {
		v := t.slots[h]
		if v == 0 {
			return 0, false
		}
		if t.str(v-1) == s {
			return v - 1, true
		}
	}
}

// lookupBytes is lookup without converting b to a string.
func (t *strTable) lookupBytes(b []byte) (int32, bool) {
	if len(b) == 0 {
		return 0, false
	}
	for h := maphash.Bytes(tableSeed, b) & t.mask; ; h = (h + 1) & t.mask {
		v := t.slots[h]
		if v == 0 {
			return 0, false
		}
		if t.str(v-1) == string(b) {
			return v - 1, true
		}
	}
}

// mergeTable finds the merge of a pair of ids by binary search in the
// sorted pairs.
type mergeTable struct {
	keys []uint64
	vals []merge
}

type mergeEntry struct {
	key uint64
	m   merge
}

// newMergeTable keeps, for a pair present several times, the merge with
// the smallest rank.
func newMergeTable(entries []mergeEntry) mergeTable {
	slices.SortStableFunc(entries, func(a, b mergeEntry) int {
		if a.key != b.key {
			if a.key < b.key {
				return -1
			}
			return 1
		}
		return int(a.m.rank - b.m.rank)
	})
	t := mergeTable{keys: make([]uint64, 0, len(entries)), vals: make([]merge, 0, len(entries))}
	for i, e := range entries {
		if i > 0 && entries[i-1].key == e.key {
			continue
		}
		t.keys = append(t.keys, e.key)
		t.vals = append(t.vals, e.m)
	}
	return t
}

func (t mergeTable) get(key uint64) (merge, bool) {
	i, ok := slices.BinarySearch(t.keys, key)
	if !ok {
		return merge{}, false
	}
	return t.vals[i], true
}
