package ch5

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"testing"
	"time"
)

func TestTwoLayerNet(t *testing.T) {
	rand.Seed(time.Now().Unix())

	learningRate := 0.1
	net := newTwoLayerNet(2, 2, 2, 0.01)

	mx := mat.NewDense(1, 2, []float64{
		1, 1,
	})

	mt := mat.NewDense(1, 2, []float64{
		1, 1,
	})


	t.Log(net.loss(mx, mt))
	t.Log(net.predict(mx))

	for i := 0; i < 1000; i++{
		gradient := net.gradient(mx, mt)
		t.Log(mat.Formatted(net.params["W1"]))
		for _, key := range []string{"W1", "b1", "W2", "b2"} {
			net.params[key].Apply(func(i, j int, v float64) float64 {
				return v - gradient[key].At(i, j)*learningRate
			}, net.params[key])
		}
	}
	t.Log(net.loss(mx, mt))
	t.Log(net.predict(mx))

}
