package modernbert

import "math"

// float32 approximations of erf and exp for inference. The standard
// library versions compute in float64 at maximum precision; in inference,
// erf and exp cost a fifth of the time. Training keeps the exact
// functions: that is what the parity tests against PyTorch check.

// erf32 is the rational approximation from Eigen and XLA: absolute error
// below 2e-7 over the whole real axis, i.e. float32 precision.
func erf32(x float32) float32 {
	x = min(max(x, -4), 4) // beyond this, erf is ±1 in float32
	x2 := x * x
	p := x2*-2.72614225801306e-10 + 2.77068142495902e-08
	p = x2*p + -2.10102402082508e-06
	p = x2*p + -5.69250639462346e-05
	p = x2*p + -7.34990630326855e-04
	p = x2*p + -2.95459980854025e-03
	p = x2*p + -1.60960333262415e-02
	p *= x
	q := x2*-1.45660718464996e-05 + -2.13374055278905e-04
	q = x2*q + -1.68282697438203e-03
	q = x2*q + -7.37332916720468e-03
	q = x2*q + -1.42647390514189e-02
	return p / q
}

func geluFast(x float32) float32 {
	return 0.5 * x * (1 + erf32(x*invSqrt2))
}

// exp32 is Cephes's expf: reduction to r ∈ [-ln2/2, ln2/2], degree-6
// polynomial, then multiplication by 2ⁿ. Relative error around 2e-7.
func exp32(x float32) float32 {
	if x < -87.3 {
		return 0 // underflow, and -Inf of masked positions
	}
	x = min(x, 88.7)
	n := float32(math.Floor(float64(x*1.44269504088896341 + 0.5)))
	r := x - n*0.693359375 - n*-2.12194440e-4
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 5.0000001201e-1
	y := p*r*r + r + 1
	return y * math.Float32frombits(uint32(int32(n)+127)<<23)
}
