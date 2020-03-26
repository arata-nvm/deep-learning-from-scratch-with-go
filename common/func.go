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
		return 1 / (1 + math.Exp(-v))
	})
}

func Relu(x mat.Matrix) mat.Matrix {
	return apply(x, func(v float64) float64 {
		return math.Max(0, v)
	})
}

func Softmax(a mat.Matrix) mat.Matrix {
	c := mat.Max(a)
	expA := apply(a, func(v float64) float64 {
		return math.Exp(v - c)
	})
	sumExpA := mat.Sum(expA)
	y := apply(expA, func(v float64) float64 {
		return v / sumExpA
	})
	return y
}

func MeanSquaredError(y, t mat.Matrix) float64 {
	r, c := y.Dims()
	diff := mat.NewDense(r, c, nil)
	diff.Sub(y, t)
	diff.MulElem(diff, diff)
	return 0.5 * mat.Sum(diff)
}

func CrossEntropyError(y, t mat.Matrix) float64 {
	delta := 1e-7
	logY := apply(y, func(v float64) float64 {
		return math.Log(v + delta)
	})

	r, c := logY.Dims()
	tLogY := mat.NewDense(r, c, nil)
	tLogY.MulElem(t, logY)

	return -mat.Sum(tLogY)
}

// TODO *mat.Dense -> mat.Matrix
func NumericalGradient(f func(v *mat.Dense) float64, x *mat.Dense) *mat.Dense {
	_, c := x.Dims()
	h := 1e-4
	grad := mat.NewDense(1, c, nil)
	for i := 0; i < c; i++ {
		tmpVal := x.At(0, i)
		x.Set(0, i, tmpVal+h)
		fxh1 := f(x)

		x.Set(0, i, tmpVal-h)
		fxh2 := f(x)

		grad.Set(0, i, (fxh1-fxh2)/(2*h))
		x.Set(0, i, tmpVal)
	}

	return grad
}

func NumericalGradientBatch(f func(v *mat.Dense) float64, x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	grad := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := mat.NewDense(1, c, x.RawRowView(i))

		rowGrad := NumericalGradient(f, row)
		rowGradArr := make([]float64, c)
		mat.Row(rowGradArr, 0, rowGrad)
		grad.SetRow(i, rowGradArr)
	}
	return grad
}
