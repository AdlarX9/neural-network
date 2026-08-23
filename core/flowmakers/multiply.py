from __future__ import annotations
from ..basics.layer import Layer
from ..utils.typing import ShapeFlow, Tensor, TensorFlow, Receive


class Multiply(Layer):
    def __init__(self: Multiply, receive: Receive = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(self: Multiply, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        reference_shape = input_shape[0]
        for shape in input_shape:
            if shape != reference_shape:
                raise ValueError("All incoming functions must have the same shape:", shape, reference_shape)
        self.output_shape = (reference_shape,)
        return self.output_shape

    def feed_forward(self: Multiply, entry: TensorFlow) -> Tensor:
        output = entry[0]
        for i in range(1, len(entry)):
            output *= entry[i]
        return output

    def descend_gradient(self: Multiply, gradient: Tensor) -> TensorFlow:
        if self.input is None:
            raise MemoryError
        gradients = [gradient.copy() for _ in range(len(self.input))]
        for i in range(len(self.input)):
            for j in range(len(self.input)):
                if j != i:
                    gradients[i] *= self.input[j]
        return tuple(gradients)
