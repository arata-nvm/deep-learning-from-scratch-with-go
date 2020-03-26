package ch4

import (
	"github.com/arata-nvm/deep-learning-from-scratch-with-go/common"
	"gonum.org/v1/gonum/mat"
)

type simpleNet struct {
	W *mat.Dense
}

func newSimpleNet() *simpleNet {
	return &simpleNet{
		W: common.Randn(2, 3),
	}
}

func (s *simpleNet) predict(x mat.Matrix) mat.Matrix {
	result := mat.NewDense(1, 3, nil)
	result.Product(x, s.W)
	return result
}

func (s *simpleNet) loss(x, t mat.Matrix) float64 {
	z := s.predict(x)
	y := common.Softmax(z)
	loss := common.CrossEntropyError(y, t)
	return loss
}
