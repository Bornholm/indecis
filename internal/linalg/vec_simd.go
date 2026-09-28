//go:build goexperiment.simd

package linalg

import (
	"math"
	"simd"
)

// Vector operations for inference, in portable SIMD (scalar fallback in
// vec_scalar.go, same contract). None is inlined: a SIMD function inlined
// into a closure crashes the Go 1.27 compiler.

// AbsMax returns max |x[i]|.
//
//go:noinline
func AbsMax(x []float32) float32 {
	var acc simd.Float32s
	V := acc.Len()
	i := 0
	for ; i+V <= len(x); i += V {
		acc = acc.Max(simd.LoadFloat32s(x[i:]).Abs())
	}
	var lanes [16]float32
	acc.Store(lanes[:])
	var m float32
	for _, v := range lanes[:V] {
		m = max(m, v)
	}
	for ; i < len(x); i++ {
		m = max(m, abs32(x[i]))
	}
	return m
}

// QuantizeRow writes dst[i] = round(x[i]*inv) + 128 (|x[i]*inv| <= 127).
//
//go:noinline
func QuantizeRow(dst []uint8, x []float32, inv float32) {
	vinv := simd.BroadcastFloat32s(inv)
	magic := simd.BroadcastFloat32s(12582912) // 1.5*2^23: round to even
	off := simd.BroadcastInt32s(128)
	V := vinv.Len()
	var tmp [16]int32
	i := 0
	for ; i+V <= len(x); i += V {
		q := simd.LoadFloat32s(x[i:]).Mul(vinv).Add(magic).Sub(magic).ConvertToInt32().Add(off)
		q.Store(tmp[:])
		d := dst[i : i+V]
		for j := range d {
			d[j] = uint8(tmp[j])
		}
	}
	for ; i < len(x); i++ {
		dst[i] = uint8(int32(roundHalfEven(x[i]*inv)) + 128)
	}
}

// Dequantize writes, for a row of C, dst[j] (+)= sa*sb[j]*(acc[j] -
// zc[j]), where zc = 128*Sum qb of a column (see MatMul8).
//
//go:noinline
func Dequantize(dst []float32, acc []int32, sb, zc []float32, sa float32, accumulate bool) {
	vs := simd.BroadcastFloat32s(sa)
	V := vs.Len()
	j := 0
	for ; j+V <= len(dst); j += V {
		a := simd.LoadInt32s(acc[j:]).ConvertToFloat32().Sub(simd.LoadFloat32s(zc[j:]))
		v := a.Mul(simd.LoadFloat32s(sb[j:])).Mul(vs)
		if accumulate {
			v = v.Add(simd.LoadFloat32s(dst[j:]))
		}
		v.Store(dst[j:])
	}
	for ; j < len(dst); j++ {
		v := sa * sb[j] * (float32(acc[j]) - zc[j])
		if accumulate {
			dst[j] += v
		} else {
			dst[j] = v
		}
	}
}

// GeluMul writes o[i] = gelu(a[i])*b[i], using the exact GELU (erf) as a
// float32 rational approximation (error < 2e-7).
//
//go:noinline
func GeluMul(o, a, b []float32) {
	half := simd.BroadcastFloat32s(0.5)
	one := simd.BroadcastFloat32s(1)
	rs2 := simd.BroadcastFloat32s(0.7071067811865476)
	lo, hi := simd.BroadcastFloat32s(-4), simd.BroadcastFloat32s(4)
	a13, a11 := simd.BroadcastFloat32s(-2.72614225801306e-10), simd.BroadcastFloat32s(2.77068142495902e-08)
	a9, a7 := simd.BroadcastFloat32s(-2.10102402082508e-06), simd.BroadcastFloat32s(-5.69250639462346e-05)
	a5, a3 := simd.BroadcastFloat32s(-7.34990630326855e-04), simd.BroadcastFloat32s(-2.95459980854025e-03)
	a1 := simd.BroadcastFloat32s(-1.60960333262415e-02)
	b8, b6 := simd.BroadcastFloat32s(-1.45660718464996e-05), simd.BroadcastFloat32s(-2.13374055278905e-04)
	b4, b2 := simd.BroadcastFloat32s(-1.68282697438203e-03), simd.BroadcastFloat32s(-7.37332916720468e-03)
	b0 := simd.BroadcastFloat32s(-1.42647390514189e-02)
	V := one.Len()
	i := 0
	for ; i+V <= len(o); i += V {
		x := simd.LoadFloat32s(a[i:])
		t := x.Mul(rs2).Max(lo).Min(hi)
		t2 := t.Mul(t)
		p := t2.MulAdd(a13, a11)
		p = t2.MulAdd(p, a9)
		p = t2.MulAdd(p, a7)
		p = t2.MulAdd(p, a5)
		p = t2.MulAdd(p, a3)
		p = t2.MulAdd(p, a1)
		p = p.Mul(t)
		q := t2.MulAdd(b8, b6)
		q = t2.MulAdd(q, b4)
		q = t2.MulAdd(q, b2)
		q = t2.MulAdd(q, b0)
		g := half.Mul(x).Mul(one.Add(p.Div(q)))
		g.Mul(simd.LoadFloat32s(b[i:])).Store(o[i:])
	}
	for ; i < len(o); i++ {
		o[i] = geluScalar(a[i]) * b[i]
	}
}

// ExpShift writes x[i] = exp(x[i] - m) (0 below -87.3, for the -Inf of
// masked positions) and returns their sum.
//
//go:noinline
func ExpShift(x []float32, m float32) float32 {
	vm := simd.BroadcastFloat32s(m)
	floor, ceil := simd.BroadcastFloat32s(-87.3), simd.BroadcastFloat32s(88.7)
	log2e := simd.BroadcastFloat32s(1.44269504088896341)
	magic := simd.BroadcastFloat32s(12582912)
	ln2hi, ln2lo := simd.BroadcastFloat32s(0.693359375), simd.BroadcastFloat32s(-2.12194440e-4)
	c5, c4 := simd.BroadcastFloat32s(1.9875691500e-4), simd.BroadcastFloat32s(1.3981999507e-3)
	c3, c2 := simd.BroadcastFloat32s(8.3334519073e-3), simd.BroadcastFloat32s(4.1665795894e-2)
	c1, c0 := simd.BroadcastFloat32s(1.6666665459e-1), simd.BroadcastFloat32s(5.0000001201e-1)
	one := simd.BroadcastFloat32s(1)
	bias := simd.BroadcastInt32s(127)
	var zero, acc simd.Float32s
	V := one.Len()
	i := 0
	for ; i+V <= len(x); i += V {
		v := simd.LoadFloat32s(x[i:]).Sub(vm)
		under := v.Less(floor)
		v = v.Max(floor).Min(ceil)
		n := v.MulAdd(log2e, magic).Sub(magic)
		r := n.MulAdd(ln2hi.Neg(), v)
		r = n.MulAdd(ln2lo.Neg(), r)
		p := r.MulAdd(c5, c4)
		p = r.MulAdd(p, c3)
		p = r.MulAdd(p, c2)
		p = r.MulAdd(p, c1)
		p = r.MulAdd(p, c0)
		y := p.Mul(r).MulAdd(r, r).Add(one)
		pow := n.ConvertToInt32().Add(bias).ShiftAllLeft(23).ToBits().BitsToFloat32()
		e := zero.IfElse(under, y.Mul(pow)) // 0 where x - m < -87.3
		e.Store(x[i:])
		acc = acc.Add(e)
	}
	var lanes [16]float32
	acc.Store(lanes[:])
	var sum float32
	for _, v := range lanes[:V] {
		sum += v
	}
	for ; i < len(x); i++ {
		e := exp32(x[i] - m)
		x[i] = e
		sum += e
	}
	return sum
}

// Scale multiplies x by s.
//
//go:noinline
func Scale(x []float32, s float32) {
	vs := simd.BroadcastFloat32s(s)
	V := vs.Len()
	i := 0
	for ; i+V <= len(x); i += V {
		simd.LoadFloat32s(x[i:]).Mul(vs).Store(x[i:])
	}
	for ; i < len(x); i++ {
		x[i] *= s
	}
}

// LayerNormRow writes o = (x - mean)/std * gamma, the moments being
// computed in float64 in blocks of V lanes.
//
//go:noinline
func LayerNormRow(o, x, gamma []float32, eps float64) {
	var acc simd.Float32s
	V := acc.Len()
	i := 0
	var sum float64
	for ; i+V <= len(x); i += V {
		acc = acc.Add(simd.LoadFloat32s(x[i:]))
	}
	var lanes [16]float32
	acc.Store(lanes[:])
	for _, v := range lanes[:V] {
		sum += float64(v)
	}
	for ; i < len(x); i++ {
		sum += float64(x[i])
	}
	mean := float32(sum / float64(len(x)))
	vm := simd.BroadcastFloat32s(mean)
	var sq simd.Float32s
	i = 0
	for ; i+V <= len(x); i += V {
		d := simd.LoadFloat32s(x[i:]).Sub(vm)
		sq = d.MulAdd(d, sq)
	}
	sq.Store(lanes[:])
	var variance float64
	for _, v := range lanes[:V] {
		variance += float64(v)
	}
	for ; i < len(x); i++ {
		d := float64(x[i] - mean)
		variance += d * d
	}
	rstd := float32(1 / math.Sqrt(variance/float64(len(x))+eps))
	vr := simd.BroadcastFloat32s(rstd)
	i = 0
	for ; i+V <= len(x); i += V {
		simd.LoadFloat32s(x[i:]).Sub(vm).Mul(vr).Mul(simd.LoadFloat32s(gamma[i:])).Store(o[i:])
	}
	for ; i < len(x); i++ {
		o[i] = (x[i] - mean) * rstd * gamma[i]
	}
}

// MaxOf returns max x[i] (-Inf for an empty slice).
//
//go:noinline
func MaxOf(x []float32) float32 {
	m := float32(math.Inf(-1))
	acc := simd.BroadcastFloat32s(m)
	V := acc.Len()
	i := 0
	for ; i+V <= len(x); i += V {
		acc = acc.Max(simd.LoadFloat32s(x[i:]))
	}
	var lanes [16]float32
	acc.Store(lanes[:])
	for _, v := range lanes[:V] {
		m = max(m, v)
	}
	for ; i < len(x); i++ {
		m = max(m, x[i])
	}
	return m
}
