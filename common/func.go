package common

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func apply(m mat.Matrix, f func(float64) float64) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		return f(v)
	}, m)
	return result
}

func Step(x mat.Matrix) mat.Matrix {
	return apply(x, func(v float64) float64 {
		if v > 0.0 {
			return 1.0
		} else {
			return 0.0
		}
	})
}

func Sigmoid(x mat.Matrix) mat.Matrix {
	return apply(x, func(v float64) float64 {
		return 1 / (1 + math.Exp(v))
	})
}

func Relu(x mat.Matrix) mat.Matrix {
	return apply(x, func(v float64) float64 {
		return math.Max(0, v)
	})
}
