from __future__ import annotations
from ..layer.layer import Layer
from ..utils.typing import ShapeFlow, Tensor, Receive1


class Transpose(Layer):
    def __init__(self: Transpose, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(
        self: Transpose,
        input_shape: ShapeFlow,
    ) -> ShapeFlow:
        super().set_input_shape(input_shape)
        if len(input_shape[0]) == 2:
            H, W = input_shape[0]
            self.output_shape = ((W, H),)
        else:
            C, H, W = input_shape[0]
            self.output_shape = ((C, W, H),)
        return self.output_shape

    def feed_forward(self: Transpose, entry: Tensor) -> Tensor:
        l = len(entry.shape)
        return entry.swapaxes(l - 2, l - 1)

    def descend_gradient(self: Transpose, gradient: Tensor) -> Tensor:
        l = len(gradient.shape)
        return gradient.swapaxes(l - 2, l - 1)
