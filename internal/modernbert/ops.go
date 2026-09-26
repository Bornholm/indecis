package modernbert

import (
	"math"

	"github.com/bornholm/indecis/internal/linalg"
)

// Granularité des boucles parallèles, en lignes.
const rowGrain = 32

// lnCache garde ce que le backward d'une LayerNorm relit.
type lnCache struct {
	xhat []float32 // (x - μ) / σ, [N, H]
	rstd []float32 // 1 / σ, [N]
}

// layerNorm calcule out = (x - μ)/σ · γ, sans biais, ligne par ligne.
func layerNorm(out, x, gamma []float32, n, h int, eps float64, c *lnCache) {
	c.xhat = grow(c.xhat, n*h)
	c.rstd = grow(c.rstd, n)
	linalg.Parallel(n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x[r*h : (r+1)*h]
			var mean float64
			for _, v := range row {
				mean += float64(v)
			}
			mean /= float64(h)
			var variance float64
			for _, v := range row {
				d := float64(v) - mean
				variance += d * d
			}
			variance /= float64(h)
			rstd := 1 / math.Sqrt(variance+eps)
			c.rstd[r] = float32(rstd)
			xh := c.xhat[r*h : (r+1)*h]
			o := out[r*h : (r+1)*h]
			for i, v := range row {
				xh[i] = float32((float64(v) - mean) * rstd)
				o[i] = xh[i] * gamma[i]
			}
		}
	})
}

// layerNormBackward ajoute dx à dxOut (accumulation : dxOut reçoit aussi le
// gradient de la connexion résiduelle) et dγ à dgamma.
func layerNormBackward(dxOut, dy, gamma, dgamma []float32, n, h int, c *lnCache) {
	// dγ se réduit sur les lignes : une somme partielle par tranche, puis
	// une réduction séquentielle, pour rester déterministe.
	parts := linalg.Partials(n, rowGrain, h, func(lo, hi int, acc []float64) {
		for r := lo; r < hi; r++ {
			dyr := dy[r*h : (r+1)*h]
			xh := c.xhat[r*h : (r+1)*h]
			for i := range dyr {
				acc[i] += float64(dyr[i]) * float64(xh[i])
			}
		}
	})
	if dgamma != nil {
		for i := range dgamma {
			dgamma[i] += float32(parts[i])
		}
	}
	linalg.Parallel(n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			dyr := dy[r*h : (r+1)*h]
			xh := c.xhat[r*h : (r+1)*h]
			var sumD, sumDX float64
			for i := range dyr {
				d := float64(dyr[i]) * float64(gamma[i])
				sumD += d
				sumDX += d * float64(xh[i])
			}
			mD, mDX := sumD/float64(h), sumDX/float64(h)
			rstd := float64(c.rstd[r])
			dx := dxOut[r*h : (r+1)*h]
			for i := range dyr {
				d := float64(dyr[i]) * float64(gamma[i])
				dx[i] += float32(rstd * (d - mD - float64(xh[i])*mDX))
			}
		}
	})
}

const invSqrt2 = 0.7071067811865476

func gelu(x float32) float32 {
	return float32(0.5 * float64(x) * (1 + math.Erf(float64(x)*invSqrt2)))
}

// geluGrad est la dérivée de la GELU exacte : Φ(x) + x·φ(x).
func geluGrad(x float32) float32 {
	xf := float64(x)
	cdf := 0.5 * (1 + math.Erf(xf*invSqrt2))
	pdf := math.Exp(-0.5*xf*xf) / math.Sqrt(2*math.Pi)
	return float32(cdf + xf*pdf)
}

// gluForward calcule g = gelu(a) ⊙ b, où z = [a | b] par ligne.
func gluForward(g, z []float32, n, inter int) {
	linalg.Parallel(n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			a := z[r*2*inter : r*2*inter+inter]
			b := z[r*2*inter+inter : (r+1)*2*inter]
			o := g[r*inter : (r+1)*inter]
			for i := range o {
				o[i] = gelu(a[i]) * b[i]
			}
		}
	})
}

// gluBackward calcule dz = [dg ⊙ b ⊙ gelu'(a) | dg ⊙ gelu(a)].
func gluBackward(dz, dg, z []float32, n, inter int) {
	linalg.Parallel(n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			a := z[r*2*inter : r*2*inter+inter]
			b := z[r*2*inter+inter : (r+1)*2*inter]
			d := dg[r*inter : (r+1)*inter]
			da := dz[r*2*inter : r*2*inter+inter]
			db := dz[r*2*inter+inter : (r+1)*2*inter]
			for i := range d {
				da[i] = d[i] * b[i] * geluGrad(a[i])
				db[i] = d[i] * gelu(a[i])
			}
		}
	})
}

// applyRope tourne x (D valeurs d'une tête à la position pos), à la façon de
// rotate_half : x' = x·cos + [-x₂, x₁]·sin.
func applyRope(x []float32, cos, sin []float32) {
	half := len(x) / 2
	for i := 0; i < half; i++ {
		x1, x2 := x[i], x[i+half]
		x[i] = x1*cos[i] - x2*sin[i]
		x[i+half] = x2*cos[i] + x1*sin[i]
	}
}

// applyRopeBackward applique la rotation transposée : c'est une rotation
// d'angle opposé, donc le gradient repasse par l'inverse exact du forward.
func applyRopeBackward(d []float32, cos, sin []float32) {
	half := len(d) / 2
	for i := 0; i < half; i++ {
		d1, d2 := d[i], d[i+half]
		d[i] = d1*cos[i] + d2*sin[i]
		d[i+half] = d2*cos[i] - d1*sin[i]
	}
}

// softmaxRows normalise chaque ligne de s (n lignes de m valeurs) en place.
// Les masques sont appliqués avant : -Inf.
func softmaxRows(s []float32, n, m int) {
	for r := 0; r < n; r++ {
		row := s[r*m : (r+1)*m]
		mx := float32(math.Inf(-1))
		for _, v := range row {
			mx = max(mx, v)
		}
		if math.IsInf(float64(mx), -1) {
			clear(row) // ligne entièrement masquée
			continue
		}
		var sum float64
		for i, v := range row {
			e := float32(math.Exp(float64(v - mx)))
			row[i] = e
			sum += float64(e)
		}
		inv := float32(1 / sum)
		for i := range row {
			row[i] *= inv
		}
	}
}

func grow(s []float32, n int) []float32 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float32, n)
}

func sqrt(x float64) float64 { return math.Sqrt(x) }
