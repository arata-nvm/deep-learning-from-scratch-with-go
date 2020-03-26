package ch3

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

var eps = math.Nextafter(1.0, 2.0) - 1.0

func TestForward(t *testing.T) {
	network := initNetwork()
	x := mat.NewDense(1, 2, []float64{
		1.0, 0.5,
	})

	expected := mat.NewDense(1, 2, []float64{
		0.3168270764110298, 0.6962790898619668,
	})

	actual := forward(network, x)
	if expected.At(0, 0)-actual.At(0, 0) > eps {
		t.Fail()
	}
	if expected.At(0, 1)-actual.At(0, 1) > eps {
		t.Fail()
	}
}
