package ch4

import (
	"github.com/arata-nvm/deep-learning-from-scratch-with-go/common"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestSimpleNet(t *testing.T) {
	mx := mat.NewDense(1, 2, []float64{
		0.6, 0.9,
	})

	mt := mat.NewDense(1, 3, []float64{
		0, 0, 1,
	})

	net := newSimpleNet()
	dw := common.NumericalGradientBatch(func(v *mat.Dense) float64 {
		return net.loss(mx, mt)
	}, net.W)

	t.Log(mat.Formatted(dw))
}
