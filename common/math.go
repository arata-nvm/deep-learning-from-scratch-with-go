package common

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

func Randn(row, col int) *mat.Dense {
	data := make([]float64, row*col)
	for i := range data {
		data[i] = rand.NormFloat64()
	}

	return mat.NewDense(row, col, data)
}

func ArgmaxR(m mat.Matrix) []int {
	r, c := m.Dims()
	result := make([]int, r)
	for i := 0; i < r; i++ {
		max := -math.MaxFloat64
		maxi := 0
		for j := 0; j < c; j++ {
			if m.At(i, j) > max {
				max = m.At(i, j)
				maxi = j
			}
		}
		result[i] = maxi
	}
	return result
}
