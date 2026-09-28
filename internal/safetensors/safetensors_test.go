package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestWriteParseRoundTrip(t *testing.T) {
	in := map[string]Tensor{
		"b": {Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, float32(math.Pi)}},
		"a": {Shape: []int{1}, Data: []float32{-0.5}},
	}
	var buf bytes.Buffer
	if err := Write(&buf, in, map[string]string{"format": "indecis"}); err != nil {
		t.Fatal(err)
	}
	out, meta, err := Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) || meta["format"] != "indecis" {
		t.Fatalf("got %v %v", out, meta)
	}
}

func TestParseBF16AndF16(t *testing.T) {
	header := `{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"F16","shape":[3],"data_offsets":[4,10]}}`
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(header)))
	b.WriteString(header)
	binary.Write(&b, binary.LittleEndian, uint16(0x3f80)) // bf16 1.0
	binary.Write(&b, binary.LittleEndian, uint16(0xc040)) // bf16 -3.0
	binary.Write(&b, binary.LittleEndian, uint16(0x3c00)) // f16 1.0
	binary.Write(&b, binary.LittleEndian, uint16(0xb800)) // f16 -0.5
	binary.Write(&b, binary.LittleEndian, uint16(0x0001)) // f16 smallest subnormal
	out, _, err := Parse(b.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out["x"].Data, []float32{1, -3}) {
		t.Fatalf("bf16: %v", out["x"].Data)
	}
	y := out["y"].Data
	if y[0] != 1 || y[1] != -0.5 || math.Abs(float64(y[2])-5.960464477539063e-08) > 1e-15 {
		t.Fatalf("f16: %v", y)
	}
}

func TestParseRejectsCorruption(t *testing.T) {
	cases := map[string][]byte{
		"short":        {1, 2, 3},
		"giant header": append(binary.LittleEndian.AppendUint64(nil, 1<<40), '{', '}'),
	}
	header := `{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}`
	bad := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	bad = append(bad, header...)
	bad = append(bad, make([]byte, 8)...)
	cases["inconsistent size"] = bad
	for name, b := range cases {
		if _, _, err := Parse(b); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestWriteBF16(t *testing.T) {
	in := map[string]Tensor{"x": {Shape: []int{3}, Data: []float32{1, -2.5, 0.1}, DType: "BF16"}}
	var buf bytes.Buffer
	if err := Write(&buf, in, nil); err != nil {
		t.Fatal(err)
	}
	out, _, err := Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	got := out["x"].Data
	if got[0] != 1 || got[1] != -2.5 || math.Abs(float64(got[2])-0.1) > 1e-3 || got[2] == 0.1 {
		t.Fatalf("got %v", got)
	}
}
