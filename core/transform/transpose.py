from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Transpose(Layer):
    def __init__(self: Transpose, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(
        self: Transpose,
        input_shape: tuple[tuple[int, int]] | tuple[tuple[int, int, int]],
    ) -> tuple[tuple[int, int]] | tuple[tuple[int, int, int]]:
        super().set_input_shape(input_shape)
        if len(input_shape[0]) == 2:
            H, W = input_shape[0]
            self.output_shape = ((W, H),)
        else:
            C, H, W = input_shape[0]
            self.output_shape = ((C, W, H),)
        return self.output_shape

    def feed_forward(self: Transpose, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        l = len(entry.shape)
        return entry.swapaxes(l - 2, l - 1)

    def descend_gradient(self: Transpose, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        l = len(gradient.shape)
        return gradient.swapaxes(l - 2, l - 1)
