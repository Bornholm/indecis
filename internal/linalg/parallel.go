package linalg

import (
	"runtime"
	"sync"
)

// Parallel découpe [0, n) en tranches d'au moins grain éléments et appelle
// fn sur chacune, en parallèle. Les tranches sont disjointes.
func Parallel(n, grain int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if grain < 1 {
		grain = 1
	}
	chunks := min(workers, (n+grain-1)/grain)
	if chunks <= 1 {
		fn(0, n)
		return
	}
	step := (n + chunks - 1) / chunks
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += step {
		wg.Add(1)
		go parallelChunk(&wg, fn, lo, min(lo+step, n))
	}
	wg.Wait()
}

func parallelChunk(wg *sync.WaitGroup, fn func(lo, hi int), lo, hi int) {
	defer wg.Done()
	fn(lo, hi)
}

// Partials réduit [0, n) par tranches fixes de grain éléments : fn accumule
// chaque tranche dans son propre vecteur de width float64, puis les vecteurs
// sont sommés dans l'ordre des tranches.
//
// Le découpage ne dépend pas du nombre de cœurs : le résultat est le même,
// au bit près, sur toutes les machines. C'est ce qui rend un entraînement
// reproductible.
func Partials(n, grain, width int, fn func(lo, hi int, acc []float64)) []float64 {
	out := make([]float64, width)
	if n <= 0 {
		return out
	}
	if grain < 1 {
		grain = 1
	}
	chunks := (n + grain - 1) / grain
	accs := make([]float64, chunks*width)
	Parallel(chunks, 1, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			fn(c*grain, min((c+1)*grain, n), accs[c*width:(c+1)*width])
		}
	})
	for c := 0; c < chunks; c++ {
		for i, v := range accs[c*width : (c+1)*width] {
			out[i] += v
		}
	}
	return out
}
