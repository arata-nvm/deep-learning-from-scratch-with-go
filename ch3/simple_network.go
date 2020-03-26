package ch3

import (
	"github.com/arata-nvm/deep-learning-from-scratch-with-go/common"
	"gonum.org/v1/gonum/mat"
)

func initNetwork() map[string]mat.Matrix {
	network := make(map[string]mat.Matrix)

	network["W1"] = mat.NewDense(2, 3, []float64{
		0.1, 0.3, 0.5,
		0.2, 0.4, 0.6,
	})
	network["b1"] = mat.NewDense(1, 3, []float64{
		0.1, 0.2, 0.3,
	})
	network["W2"] = mat.NewDense(3, 2, []float64{
		0.1, 0.4,
		0.2, 0.5,
		0.3, 0.6,
	})
	network["b2"] = mat.NewDense(1, 2, []float64{
		0.1, 0.2,
	})
	network["W3"] = mat.NewDense(2, 2, []float64{
		0.1, 0.3,
		0.2, 0.4,
	})
	network["b3"] = mat.NewDense(1, 2, []float64{
		0.1, 0.2,
	})

	return network
}

func forward(network map[string]mat.Matrix, x mat.Matrix) mat.Matrix {
	W1, W2, W3 := network["W1"], network["W2"], network["W3"]
	b1, b2, b3 := network["b1"], network["b2"], network["b3"]

	a1 := mat.NewDense(1, 3, nil)
	a1.Product(x, W1)
	a1.Add(a1, b1)
	z1 := common.Sigmoid(a1)

	a2 := mat.NewDense(1, 2, nil)
	a2.Product(z1, W2)
	a2.Add(a2, b2)
	z2 := common.Sigmoid(a2)

	a3 := mat.NewDense(1, 2, nil)
	a3.Product(z2, W3)
	a3.Add(a3, b3)
	y := a3

	return y
}
