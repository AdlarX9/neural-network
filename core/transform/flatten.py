from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..utils.typing import ShapeFlow, Tensor, Receive1
from ..layer.layer import Layer


class Flatten(Layer):
    def __init__(self: Flatten, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(self: Flatten, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        c, n, p = input_shape[0]
        self.output_shape = ((c * n * p, 1),)
        return self.output_shape

    def feed_forward(self: Flatten, entry: Tensor) -> Tensor:
        return entry.reshape(-1, 1)

    def descend_gradient(self: Flatten, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        return gradient.reshape(self.input[0].shape)
