package ch4

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestTwoLayerNet(t *testing.T) {
	net := newTwoLayerNet(2, 50, 2, 0.01)

	net.numericalGradient(
		mat.NewDense(1, 2, []float64{
			1, 1,
		}),
		mat.NewDense(1, 2, []float64{
			1, 1,
		}),
	)
}
