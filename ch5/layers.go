package ch5

type mulLayer struct {
	x float64
	y float64
}

func newMulLayer() *mulLayer {
	return &mulLayer{}
}

func (l *mulLayer) forward(x, y float64) float64 {
	l.x = x
	l.y = y

	return x * y
}

func (l *mulLayer) backward(dout float64) (float64, float64) {
	dx := dout * l.y
	dy := dout * l.x

	return dx, dy
}

type addLayer struct {
}

func newAddLayer() *addLayer {
	return &addLayer{}
}

func (l *addLayer) forward(x, y float64) float64 {
	return x + y
}

func (l *addLayer) backward(dout float64) (float64, float64) {
	dx := dout * 1
	dy := dout * 1

	return dx, dy
}
