package common

import "gonum.org/v1/gonum/mat"

func Step(x mat.Matrix) mat.Matrix {
	step := func(i, j int, v float64) float64 {
		if v > 0.0 {
			return 1.0
		} else {
			return 0.0
		}
	}

	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(step, x)
	return result
}
