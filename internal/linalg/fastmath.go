package linalg

import "math"

// float32 approximations shared by the SIMD and scalar versions.

// erf32 is the rational approximation from Eigen and XLA: absolute error
// below 2e-7 over the whole real axis.
func erf32(x float32) float32 {
	x = min(max(x, -4), 4)
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

func geluScalar(x float32) float32 { return 0.5 * x * (1 + erf32(x*0.7071067811865476)) }

// geluTanhScalar is the tanh approximation of GELU (PyTorch's "tanh" GELU),
// written x * sigmoid(2u) with u = sqrt(2/pi) * (x + 0.044715 x^3).
func geluTanhScalar(x float32) float32 {
	u := 0.7978845608028654 * (x + 0.044715*x*x*x)
	return x / (1 + exp32(-2*u))
}

// exp32 is Cephes' expf (relative error ~ 2e-7); 0 below -87.3.
func exp32(x float32) float32 {
	if x < -87.3 {
		return 0
	}
	x = min(x, 88.7)
	n := roundHalfEven(x * 1.44269504088896341)
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
