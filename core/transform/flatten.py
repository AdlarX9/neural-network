from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class Flatten(Layer):
    def __init__(self: Flatten) -> None:
        super().__init__()

    def set_input_shape(self: Flatten, input_shape: tuple[int, int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        c, n, p = input_shape
        self.output_shape = (c * n * p, 1)
        return self.output_shape

    def feed_forward(self: Flatten, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry.reshape(-1, 1)

    def descend_gradient(self: Flatten, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        return gradient.reshape(self.input.shape)
