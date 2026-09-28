package tokenizer

import (
	"encoding/json"
	"fmt"
	"sort"
)

// Prune removes from a tokenizer.json the tokens that keep rejects and
// returns the new file, along with the id mapping: newID[old] is the new
// id, or -1 for a removed token.
//
// Always kept: added tokens (including special ones), the fallback byte
// tokens (<0x00>...<0xFF>) and the unknown token, without which some texts
// would no longer split. A merge is kept only if both its parts and its
// result are. Remaining ids are renumbered in their original order, with
// no gaps.
//
// A text whose split (Trace) only goes through kept tokens splits exactly
// as before, aside from the ids. Another one splits into smaller pieces,
// down to bytes in the worst case: the result stays valid, but the model
// did not see it during training.
func Prune(src []byte, keep func(id int32) bool) ([]byte, []int32, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(src, &root); err != nil {
		return nil, nil, fmt.Errorf("tokenizer: %w", err)
	}
	var model map[string]json.RawMessage
	if err := json.Unmarshal(root["model"], &model); err != nil {
		return nil, nil, fmt.Errorf("tokenizer: model : %w", err)
	}
	var vocab map[string]int32
	if err := json.Unmarshal(model["vocab"], &vocab); err != nil {
		return nil, nil, fmt.Errorf("tokenizer: vocab : %w", err)
	}
	var added []map[string]any
	if err := json.Unmarshal(root["added_tokens"], &added); err != nil {
		return nil, nil, fmt.Errorf("tokenizer: added_tokens : %w", err)
	}
	var unk string
	json.Unmarshal(model["unk_token"], &unk)

	size := int32(0)
	for _, id := range vocab {
		size = max(size, id+1)
	}
	forced := map[int32]bool{}
	for _, a := range added {
		id := int32(a["id"].(float64))
		forced[id] = true
		size = max(size, id+1)
	}
	for tok, id := range vocab {
		if tok == unk || isByteToken(tok) {
			forced[id] = true
		}
	}

	newID := make([]int32, size)
	next := int32(0)
	for id := int32(0); id < size; id++ {
		if forced[id] || keep(id) {
			newID[id] = next
			next++
		} else {
			newID[id] = -1
		}
	}

	kept := make(map[string]int32, next)
	for tok, id := range vocab {
		if newID[id] >= 0 {
			kept[tok] = newID[id]
		}
	}
	for _, a := range added {
		a["id"] = newID[int32(a["id"].(float64))]
	}

	pairs, err := decodeMergesJSON(model["merges"])
	if err != nil {
		return nil, nil, err
	}
	var merges [][2]string
	for _, p := range pairs {
		_, okA := kept[p[0]]
		_, okB := kept[p[1]]
		_, okM := kept[p[0]+p[1]]
		if okA && okB && okM {
			merges = append(merges, p)
		}
	}

	if model["vocab"], err = marshalVocab(kept); err != nil {
		return nil, nil, err
	}
	if model["merges"], err = json.Marshal(merges); err != nil {
		return nil, nil, err
	}
	if root["model"], err = json.Marshal(model); err != nil {
		return nil, nil, err
	}
	if root["added_tokens"], err = json.Marshal(added); err != nil {
		return nil, nil, err
	}
	if pp, ok := root["post_processor"]; ok && string(pp) != "null" {
		if root["post_processor"], err = remapPostProcessor(pp, newID); err != nil {
			return nil, nil, err
		}
	}
	out, err := json.Marshal(root)
	if err != nil {
		return nil, nil, err
	}
	if _, err := Parse(out); err != nil {
		return nil, nil, fmt.Errorf("tokenizer: pruned file is invalid: %w", err)
	}
	return out, newID, nil
}

func isByteToken(tok string) bool {
	return len(tok) == 6 && tok[:3] == "<0x" && tok[5] == '>'
}

// marshalVocab writes the vocabulary in id order, like Hugging Face
// files do.
func marshalVocab(v map[string]int32) (json.RawMessage, error) {
	type entry struct {
		tok string
		id  int32
	}
	entries := make([]entry, 0, len(v))
	for t, id := range v {
		entries = append(entries, entry{t, id})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].id < entries[j].id })
	buf := []byte{'{'}
	for i, e := range entries {
		if i > 0 {
			buf = append(buf, ',')
		}
		k, err := json.Marshal(e.tok)
		if err != nil {
			return nil, err
		}
		buf = append(buf, k...)
		buf = append(buf, ':')
		buf = fmt.Appendf(buf, "%d", e.id)
	}
	return append(buf, '}'), nil
}

func decodeMergesJSON(raw json.RawMessage) ([][2]string, error) {
	var pairs [][2]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		return pairs, nil
	}
	var legacy []string
	if err := json.Unmarshal(raw, &legacy); err != nil {
		return nil, fmt.Errorf("tokenizer: merges : %w", err)
	}
	pairs = make([][2]string, 0, len(legacy))
	for _, s := range legacy {
		for i := 0; i < len(s); i++ {
			if s[i] == ' ' {
				pairs = append(pairs, [2]string{s[:i], s[i+1:]})
				break
			}
		}
	}
	return pairs, nil
}

// remapPostProcessor renumbers the special token ids in the template.
func remapPostProcessor(raw json.RawMessage, newID []int32) (json.RawMessage, error) {
	var pp map[string]any
	if err := json.Unmarshal(raw, &pp); err != nil {
		return nil, fmt.Errorf("tokenizer: post_processor : %w", err)
	}
	if st, ok := pp["special_tokens"].(map[string]any); ok {
		for _, v := range st {
			tok, ok := v.(map[string]any)
			if !ok {
				continue
			}
			ids, _ := tok["ids"].([]any)
			for i, x := range ids {
				if f, ok := x.(float64); ok && int(f) < len(newID) {
					ids[i] = newID[int(f)]
				}
			}
		}
	}
	return json.Marshal(pp)
}
