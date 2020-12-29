package main

import (
	"fmt"
	. "github.com/arata-nvm/deep-learning-from-scratch-with-go/common"
	"gonum.org/v1/gonum/mat"
)

func main() {
	x := mat.NewDense(1, 2, []float64{
		0.5, 1,
	})
	t := mat.NewDense(1, 2, []float64{
		1, 0.5,
	})

	fmt.Println(mat.Formatted(Softmax(x)))
	fmt.Println(CrossEntropyError(Softmax(x), t))

	l := NewSoftmaxWithLossLayer()
	y := l.Forward(x, t)
	fmt.Println(mat.Formatted(y))

	dy := l.Backward(x)
	fmt.Println(mat.Formatted(dy))
}
