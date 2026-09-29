package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// The hardware kernel and the portable kernel produce the same integers.
func TestMicroKernel8MatchesGo(t *testing.T) {
	if !Int8Fast() {
		t.Skip("no AVX-VNNI")
	}
	r := rand.New(rand.NewSource(1))
	for _, kq := range []int{1, 3, 96, 288} {
		lda := kq*4 + 12 // not just the width of the row: the kernel must respect it
		ap := make([]uint8, mr8*lda)
		bp := make([]int8, kq*nr8*4)
		for i := range ap {
			ap[i] = uint8(r.Intn(256))
		}
		for i := range bp {
			bp[i] = int8(r.Intn(255) - 127)
		}
		var got, want [mr8 * nr8]int32
		microKernel8(kq, ap, lda, bp, &got)
		microKernel8Go(kq, ap, lda, bp, &want)
		if got != want {
			t.Fatalf("kq=%d: %v\nwant %v", kq, got[:8], want[:8])
		}
	}
}

// The quantized product stays close to the float32 product: the error
// comes from int8 rounding, not from packing.
func TestMatMul8(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, s := range [][3]int{{1, 1, 1}, {7, 384, 1152}, {15, 1152, 384}, {33, 301, 17}} {
		m, k, n := s[0], s[1], s[2]
		for _, transB := range []bool{false, true} {
			a := randSlice(r, m*k)
			b := randSlice(r, k*n)
			want := make([]float32, m*n)
			MatMul(want, a, b, m, k, n, false, transB, false)
			p := PackB8(b, k, n, transB)
			got := make([]float32, m*n)
			MatMul8(got, a, p, m, false)
			var num, den float64
			for i := range got {
				d := float64(got[i] - want[i])
				num += d * d
				den += float64(want[i]) * float64(want[i])
			}
			if rel := math.Sqrt(num / den); rel > 0.02 {
				t.Fatalf("%v transB=%v: relative error %.3g", s, transB, rel)
			}
			before := append([]float32(nil), got...)
			MatMul8(got, a, p, m, true)
			for i := range got {
				if math.Abs(float64(got[i]-2*before[i])) > 1e-4*math.Abs(float64(before[i]))+1e-6 {
					t.Fatalf("%v accumulate: [%d] %v, want %v", s, i, got[i], 2*before[i])
				}
			}
		}
	}
}

func BenchmarkMatMul8(b *testing.B) {
	r := rand.New(rand.NewSource(3))
	m, k, n := 28, 384, 2304
	a := randSlice(r, m*k)
	w := randSlice(r, k*n)
	c := make([]float32, m*n)
	p8 := PackB8(w, k, n, true)
	p32 := PackB(w, k, n, true)
	b.Run("int8", func(b *testing.B) {
		for range b.N {
			MatMul8(c, a, p8, m, false)
		}
	})
	b.Run("float32", func(b *testing.B) {
		for range b.N {
			MatMulPacked(c, a, p32, m, false)
		}
	})
}

func TestRoundHalfEven(t *testing.T) {
	for _, v := range []float32{0, 0.5, 1.5, 2.5, -0.5, -1.5, 126.7, -127, 3.49999, -3.5000002} {
		if got, want := roundHalfEven(v), float32(math.RoundToEven(float64(v))); got != want {
			t.Errorf("roundHalfEven(%v) = %v, want %v", v, got, want)
		}
	}
}

// With a few massive input channels, splitting them off keeps the int8
// product close to float32, where per-row quantization loses the rest.
func TestMatMul8Outliers(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	m, k, n := 64, 3072, 96
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	for i := 0; i < m; i++ {
		a[i*k+1038] = 1000 * float32(r.NormFloat64())
		a[i*k+7] = 150
	}
	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(r.NormFloat64() * 0.02)
	}
	want := make([]float32, m*n)
	MatMul(want, a, b, m, k, n, false, false, false)
	p := PackB8(b, k, n, false)
	plain, split := make([]float32, m*n), make([]float32, m*n)
	MatMul8(plain, a, p, m, false)
	MatMul8Outliers(split, a, p, m, false, 6, 0)
	errOf := func(got []float32) float64 {
		var e, s float64
		for i := range want {
			d := float64(got[i] - want[i])
			e += d * d
			s += float64(want[i]) * float64(want[i])
		}
		return math.Sqrt(e / s)
	}
	ep, es := errOf(plain), errOf(split)
	t.Logf("relative error: MatMul8 %.2e, MatMul8Outliers %.2e", ep, es)
	if es > ep/5 || es > 1e-2 {
		t.Fatalf("MatMul8Outliers error %.2e, MatMul8 %.2e", es, ep)
	}
	// Without outliers, the same result as MatMul8.
	for i := 0; i < m; i++ {
		a[i*k+1038], a[i*k+7] = 0, 0
	}
	MatMul8(plain, a, p, m, false)
	MatMul8Outliers(split, a, p, m, false, 6, 0)
	for i := range plain {
		if plain[i] != split[i] {
			t.Fatalf("without outliers, [%d] %v vs %v", i, split[i], plain[i])
		}
	}
}
