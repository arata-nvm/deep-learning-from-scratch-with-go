package ch2

import "gonum.org/v1/gonum/mat"

func And(x1, x2 float64) float64 {
	x := mat.NewVecDense(2, []float64{x1, x2})
	w := mat.NewVecDense(2, []float64{0.5, 0.5})
	b := -0.7
	x.MulElemVec(x, w)
	tmp := mat.Sum(x) + b
	if tmp <= 0 {
		return 0
	} else {
		return 1
	}
}
