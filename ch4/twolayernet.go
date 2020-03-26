package ch4

import (
	"github.com/arata-nvm/deep-learning-from-scratch-with-go/common"
	"gonum.org/v1/gonum/mat"
)

type twoLayerNet struct {
	params map[string]*mat.Dense

	inputSize  int
	hiddenSize int
	outputSize int
}

func newTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *twoLayerNet {
	params := make(map[string]*mat.Dense)
	params["W1"] = common.Randn(inputSize, hiddenSize)
	params["W1"].Scale(weightInitStd, params["W1"])
	params["b1"] = mat.NewDense(1, hiddenSize, nil)
	params["W2"] = common.Randn(hiddenSize, outputSize)
	params["W2"].Scale(weightInitStd, params["W2"])
	params["b2"] = mat.NewDense(1, outputSize, nil)

	return &twoLayerNet{
		params:     params,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
	}
}

func (n *twoLayerNet) predict(x mat.Matrix) mat.Matrix {
	W1, W2 := n.params["W1"], n.params["W2"]
	b1, b2 := n.params["b1"], n.params["b2"]

	a1 := mat.NewDense(1, n.hiddenSize, nil)
	a1.Product(x, W1)
	a1.Add(a1, b1)
	z1 := common.Sigmoid(a1)

	a2 := mat.NewDense(1, n.outputSize, nil)
	a2.Product(z1, W2)
	a2.Add(a2, b2)
	y := common.Softmax(a2)

	return y
}

func (n *twoLayerNet) loss(x, t mat.Matrix) float64 {
	y := n.predict(x)
	return common.CrossEntropyError(y, t)
}

func (n *twoLayerNet) accuracy(x, t mat.Matrix) float64 {
	r, _ := x.Dims()

	y := n.predict(x)
	ya := common.ArgmaxR(y)
	ta := common.ArgmaxR(t)

	sum := 0
	for i := 0; i < r; i++ {
		if ya[i] == ta[i] {
			sum++
		}
	}

	return float64(sum) / float64(r)
}

func (n *twoLayerNet) numericalGradient(x, t mat.Matrix) map[string]mat.Matrix {
	lossW := func(v *mat.Dense) float64 {
		return n.loss(x, t)
	}

	grads := map[string]mat.Matrix{
		"W1": common.NumericalGradient(lossW, n.params["W1"]),
		"b1": common.NumericalGradient(lossW, n.params["b1"]),
		"W2": common.NumericalGradient(lossW, n.params["W2"]),
		"b2": common.NumericalGradient(lossW, n.params["b2"]),
	}

	return grads
}
