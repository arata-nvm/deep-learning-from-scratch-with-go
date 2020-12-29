package ch5

import (
	"fmt"
	"github.com/arata-nvm/deep-learning-from-scratch-with-go/common"
	"gonum.org/v1/gonum/mat"
)

type twoLayerNet struct {
	params    map[string]*mat.Dense
	layers    []common.Layer
	lastLayer *common.SoftmaxWithLossLayer

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

	var layers []common.Layer
	layers = append(layers, common.NewAffineLayer(params["W1"], params["b1"]))
	layers = append(layers, common.NewReluLayer())
	layers = append(layers, common.NewAffineLayer(params["W2"], params["b2"]))
	lastLayer := common.NewSoftmaxWithLossLayer()

	return &twoLayerNet{
		params:     params,
		layers:     layers,
		lastLayer:  lastLayer,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
	}
}

func (n *twoLayerNet) predict(x mat.Matrix) mat.Matrix {
	for _, l := range n.layers {
		x = l.Forward(x)
	}

	return x
}

func (n *twoLayerNet) loss(x, t mat.Matrix) float64 {
	y := n.predict(x)
	return n.lastLayer.Forward(y, t).At(0, 0) // TODO
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

func (n *twoLayerNet) gradient(x, t mat.Matrix) map[string]mat.Matrix {
	n.loss(x, t)

	var dout mat.Matrix
	dout = n.lastLayer.Backward(dout)

	for i := range n.layers {
		dout = n.layers[len(n.layers)-i-1].Backward(dout)
	}

	cnt := 1
	grads := make(map[string]mat.Matrix)
	for i := range n.layers {
		if a, ok := n.layers[i].(*common.AffineLayer); ok {
			wn := fmt.Sprintf("W%d", cnt)
			bn := fmt.Sprintf("b%d", cnt)
			cnt++

			grads[wn] = a.DW
			grads[bn] = a.DB
		}
	}

	return grads
}
