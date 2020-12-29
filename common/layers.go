package common

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type Layer interface {
	Forward(mat.Matrix) mat.Matrix
	Backward(mat.Matrix) mat.Matrix
}

type ReluLayer struct {
	Mask []bool
}

func NewReluLayer() *ReluLayer {
	return &ReluLayer{}
}

func (l *ReluLayer) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	l.Mask = make([]bool, r*c)

	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		if v <= 0 {
			l.Mask[r*i+j] = true
		}
		return math.Max(0, v)
	}, x)

	return result
}

func (l *ReluLayer) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		if l.Mask[r*i+j] {
			return 0
		} else {
			return v
		}
	}, dout)

	return result
}

type SigmoidLayer struct {
	Out mat.Matrix
}

func NewSigmoidLayer() *SigmoidLayer {
	return &SigmoidLayer{}
}

func (l *SigmoidLayer) Forward(x mat.Matrix) mat.Matrix {
	l.Out = Sigmoid(x)
	return l.Out
}

func (l *SigmoidLayer) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		out := l.Out.At(i, j)
		return v * (1 - out) * out
	}, dout)
	return result
}

type AffineLayer struct {
	W  mat.Matrix
	B  mat.Matrix
	X  mat.Matrix
	DW mat.Matrix
	DB mat.Matrix
}

func NewAffineLayer(w, b mat.Matrix) *AffineLayer {
	return &AffineLayer{
		W: w,
		B: b,
	}
}

func (l *AffineLayer) Forward(x mat.Matrix) mat.Matrix {
	l.X = x
	r, _ := x.Dims()
	_, c := l.W.Dims()
	out := mat.NewDense(r, c, nil)
	out.Product(x, l.W)
	out.Add(out, l.B)
	return out
}

func (l *AffineLayer) Backward(dout mat.Matrix) mat.Matrix {
	wt := l.W.T()
	r, _ := dout.Dims()
	_, c := wt.Dims()
	out := mat.NewDense(r, c, nil)
	out.Product(dout, wt)

	xt := l.X.T()
	r, _ = xt.Dims()
	_, c = dout.Dims()
	dw := mat.NewDense(r, c, nil)
	dw.Product(xt, dout)
	l.DW = dw

	r, c = dout.Dims()
	db := mat.NewDense(1, c, nil)
	for i := 0; i < c; i++ {
		sum := 0.0
		for j := 0; j < r; j++ {
			sum += dout.At(j, i)
		}
		db.Set(0, i, sum)
	}
	l.DB = db

	return out
}

type SoftmaxWithLossLayer struct {
	Loss float64
	Y    mat.Matrix
	T    mat.Matrix
}

func NewSoftmaxWithLossLayer() *SoftmaxWithLossLayer {
	return &SoftmaxWithLossLayer{}
}

func (l *SoftmaxWithLossLayer) Forward(x, t mat.Matrix) mat.Matrix {
	l.T = t
	l.Y = Softmax(x)
	l.Loss = CrossEntropyError(l.Y, l.T)
	return mat.NewDense(1, 1, []float64{l.Loss})
}

func (l *SoftmaxWithLossLayer) Backward(_ mat.Matrix) mat.Matrix {
	r, c := l.Y.Dims()
	result := mat.NewDense(r, c, nil)
	result.Sub(l.Y, l.T)
	return result
}
