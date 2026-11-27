package linalg

import (
	"math"
	"testing"
)

// BenchmarkTransition measures the AVX/SSE transition penalty: a vector
// operation from vec_simd.go, then scalar float code (float64 math.Exp,
// float32 exp32), with and without ZeroUpper in between.
func BenchmarkTransition(b *testing.B) {
	x := make([]float32, 384)
	for i := range x {
		x[i] = float32(i%17) * 0.1
	}
	var sink float64
	scalar64 := func() {
		var s float64
		for _, v := range x {
			s += math.Exp(float64(v) * 0.01)
		}
		sink += s
	}
	scalar32 := func() {
		var s float32
		for _, v := range x {
			s += exp32(v * 0.01)
		}
		sink += float64(s)
	}
	for _, c := range []struct {
		name   string
		scalar func()
		zero   bool
	}{
		{"math.Exp/dirty", scalar64, false},
		{"math.Exp/ZeroUpper", scalar64, true},
		{"exp32/dirty", scalar32, false},
		{"exp32/ZeroUpper", scalar32, true},
	} {
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				Scale(x, 1)
				if c.zero {
					ZeroUpper()
				}
				c.scalar()
			}
		})
	}
	if sink == 0 {
		b.Fatal("no work")
	}
}

// The vector operations whose scalar reductions follow vector code.
func BenchmarkReductions(b *testing.B) {
	x := make([]float32, 384)
	g := make([]float32, 384)
	for i := range x {
		x[i], g[i] = float32(i%17)*0.1, 1
	}
	o := make([]float32, 384)
	b.Run("LayerNormRow/384", func(b *testing.B) {
		for b.Loop() {
			LayerNormRow(o, x, g, 1e-5)
		}
	})
	b.Run("MaxOf/128", func(b *testing.B) {
		for b.Loop() {
			o[0] = MaxOf(x[:128])
		}
	})
	b.Run("AbsMax/128", func(b *testing.B) {
		for b.Loop() {
			o[0] = AbsMax(x[:128])
		}
	})
	q := make([]uint8, 384)
	b.Run("QuantizeRowU8/100", func(b *testing.B) {
		for b.Loop() {
			QuantizeRowU8(q[:100], x[:100], 100)
		}
	})
	b.Run("ExpShift/100", func(b *testing.B) {
		for b.Loop() {
			copy(o, x[:100])
			ExpShift(o[:100], 1)
		}
	})
}
