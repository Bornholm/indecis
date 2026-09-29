package linalg

import (
	"math"
	"math/rand"
	"testing"
)

func TestVectorOps(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 7, 16, 37, 384, 1153} {
		x := make([]float32, n)
		b := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64() * 3)
			b[i] = float32(r.NormFloat64())
		}
		var want float32
		for _, v := range x {
			want = max(want, float32(math.Abs(float64(v))))
		}
		if got := AbsMax(x); got != want {
			t.Fatalf("n=%d AbsMax %v, want %v", n, got, want)
		}

		q := make([]uint8, n)
		inv := 127 / want
		QuantizeRow(q, x, inv)
		for i, v := range x {
			if w := uint8(int32(math.RoundToEven(float64(v*inv))) + 128); q[i] != w {
				t.Fatalf("n=%d QuantizeRow[%d] %d, want %d", n, i, q[i], w)
			}
		}

		o := make([]float32, n)
		GeluMul(o, x, b)
		for i := range o {
			g := 0.5 * float64(x[i]) * (1 + math.Erf(float64(x[i])/math.Sqrt2)) * float64(b[i])
			if math.Abs(float64(o[i])-g) > 1e-5*(1+math.Abs(g)) {
				t.Fatalf("n=%d GeluMul[%d] %v, want %v", n, i, o[i], g)
			}
		}

		GeluTanh(o, x)
		for i := range o {
			xv := float64(x[i])
			g := 0.5 * xv * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(xv+0.044715*xv*xv*xv)))
			if math.Abs(float64(o[i])-g) > 2e-6*(1+math.Abs(g)) {
				t.Fatalf("n=%d GeluTanh[%d] %v, want %v", n, i, o[i], g)
			}
		}

		s := append([]float32(nil), x...)
		AddTo(s, b)
		for i := range s {
			if s[i] != x[i]+b[i] {
				t.Fatalf("n=%d AddTo[%d] %v, want %v", n, i, s[i], x[i]+b[i])
			}
		}

		e := append([]float32(nil), x...)
		e[0] = float32(math.Inf(-1))
		m := float32(2)
		sum := ExpShift(e, m)
		var ws float64
		for i, v := range x {
			w := math.Exp(float64(v - m))
			if i == 0 {
				w = 0
			}
			ws += w
			if math.Abs(float64(e[i])-w) > 1e-6*(1+w) {
				t.Fatalf("n=%d ExpShift[%d] %v, want %v", n, i, e[i], w)
			}
		}
		if math.Abs(float64(sum)-ws) > 1e-5*ws+1e-7 {
			t.Fatalf("n=%d sum %v, want %v", n, sum, ws)
		}

		gamma := b
		LayerNormRow(o, x, gamma, 1e-5)
		var mean, v2 float64
		for _, v := range x {
			mean += float64(v)
		}
		mean /= float64(n)
		for _, v := range x {
			v2 += (float64(v) - mean) * (float64(v) - mean)
		}
		rstd := 1 / math.Sqrt(v2/float64(n)+1e-5)
		for i := range o {
			w := (float64(x[i]) - mean) * rstd * float64(gamma[i])
			if math.Abs(float64(o[i])-w) > 1e-5*(1+math.Abs(w)) {
				t.Fatalf("n=%d LayerNormRow[%d] %v, want %v", n, i, o[i], w)
			}
		}

		acc := make([]int32, n)
		zc := make([]float32, n)
		for i := range acc {
			acc[i] = int32(r.Intn(200000) - 100000)
			zc[i] = float32(128 * (r.Intn(2000) - 1000))
		}
		Dequantize(o, acc, b, zc, 0.01, false)
		for i := range o {
			w := 0.01 * float64(b[i]) * (float64(acc[i]) - float64(zc[i]))
			if math.Abs(float64(o[i])-w) > 1e-5*(1+math.Abs(w)) {
				t.Fatalf("n=%d Dequantize[%d] %v, want %v", n, i, o[i], w)
			}
		}
	}
}
