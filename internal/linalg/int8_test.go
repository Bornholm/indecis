package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// Le noyau matériel et le noyau portable donnent les mêmes entiers.
func TestMicroKernel8MatchesGo(t *testing.T) {
	if !Int8Fast() {
		t.Skip("pas d'AVX-VNNI")
	}
	r := rand.New(rand.NewSource(1))
	for _, kq := range []int{1, 3, 96, 288} {
		lda := kq*4 + 12 // pas plus large que la ligne : le noyau doit le respecter
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
			t.Fatalf("kq=%d : %v\nattendu %v", kq, got[:8], want[:8])
		}
	}
}

// Le produit quantifié reste proche du produit float32 : l'erreur tient
// aux arrondis int8, pas à l'empaquetage.
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
				t.Fatalf("%v transB=%v : erreur relative %.3g", s, transB, rel)
			}
			before := append([]float32(nil), got...)
			MatMul8(got, a, p, m, true)
			for i := range got {
				if math.Abs(float64(got[i]-2*before[i])) > 1e-4*math.Abs(float64(before[i]))+1e-6 {
					t.Fatalf("%v accumulate : [%d] %v, attendu %v", s, i, got[i], 2*before[i])
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
			t.Errorf("roundHalfEven(%v) = %v, attendu %v", v, got, want)
		}
	}
}
