package ch2

import (
	"testing"
)

var input = []struct {
	x1 float64
	x2 float64
}{
	{0, 0},
	{1, 0},
	{0, 1},
	{1, 1},
}

func TestAnd(t *testing.T) {
	expected := []float64{
		0.0,
		0.0,
		0.0,
		1.0,
	}

	for i, d := range input {
		actual := And(d.x1, d.x2)
		if actual != expected[i] {
			t.Fail()
		}
	}
}

func TestNand(t *testing.T) {
	expected := []float64{
		1.0,
		1.0,
		1.0,
		0.0,
	}

	for i, d := range input {
		actual := Nand(d.x1, d.x2)
		if actual != expected[i] {
			t.Fail()
		}
	}
}
