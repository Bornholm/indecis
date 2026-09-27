//go:build !goexperiment.simd

package linalg

import "math"

// Versions scalaires des opérations vectorielles (voir vec_simd.go).

func AbsMax(x []float32) float32 {
	var m float32
	for _, v := range x {
		m = max(m, abs32(v))
	}
	return m
}

func QuantizeRow(dst []uint8, x []float32, inv float32) {
	for i, v := range x {
		dst[i] = uint8(int32(roundHalfEven(v*inv)) + 128)
	}
}

func Dequantize(dst []float32, acc []int32, sb, zc []float32, sa float32, accumulate bool) {
	for j := range dst {
		v := sa * sb[j] * (float32(acc[j]) - zc[j])
		if accumulate {
			dst[j] += v
		} else {
			dst[j] = v
		}
	}
}

func GeluMul(o, a, b []float32) {
	for i := range o {
		o[i] = geluScalar(a[i]) * b[i]
	}
}

func ExpShift(x []float32, m float32) float32 {
	var sum float32
	for i, v := range x {
		e := exp32(v - m)
		x[i] = e
		sum += e
	}
	return sum
}

func Scale(x []float32, s float32) {
	for i := range x {
		x[i] *= s
	}
}

func LayerNormRow(o, x, gamma []float32, eps float64) {
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := float32(sum / float64(len(x)))
	var variance float64
	for _, v := range x {
		d := float64(v - mean)
		variance += d * d
	}
	rstd := float32(1 / math.Sqrt(variance/float64(len(x))+eps))
	for i, v := range x {
		o[i] = (v - mean) * rstd * gamma[i]
	}
}

func MaxOf(x []float32) float32 {
	m := float32(math.Inf(-1))
	for _, v := range x {
		m = max(m, v)
	}
	return m
}
