from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Activation(Layer):
    def __init__(self: Activation, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)

    def feed_forward(self: Activation, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry

    def compute_derivative(self: Activation, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(entry)

    def descend_gradient(self: Activation, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        return gradient * self.compute_derivative(self.input[0])
