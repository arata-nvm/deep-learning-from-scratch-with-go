package ch5

import "testing"

func TestLayer(t *testing.T) {
	apple := 100.0
	appleNum := 2.0
	tax := 1.1

	mulAppleLayer := newMulLayer()
	mulTaxLayer := newMulLayer()

	applePrice := mulAppleLayer.forward(apple, appleNum)
	price := mulTaxLayer.forward(applePrice, tax)

	dPrice := 1.0
	dApplePrice, dTax := mulTaxLayer.backward(dPrice)
	dApple, dAppleNum := mulAppleLayer.backward(dApplePrice)

	t.Error(dApple, dAppleNum, dTax)

	if price != 220.0 {
		t.Fail()
	}
}
