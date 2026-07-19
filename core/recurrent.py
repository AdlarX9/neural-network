from __future__ import annotations
from .layer import Layer
from .block import Block
from .lstm import LSTM
import numpy as np
from numpy.typing import NDArray


class Recurrent(Block):
    def __init__(self: Recurrent, *layers: Layer):
        super().__init__(*layers)

    def reset_data(self: Recurrent) -> None:
        for layer in self.layers:
            if isinstance(layer, LSTM):
                layer.reset_data()

    def set_input_shape(self: Recurrent, input_shape: tuple[int, int]) -> tuple:
        n, _ = input_shape
        self.input_shape = input_shape
        return super().set_input_shape((n, 1))

    def compute(self: Recurrent, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        _, p = entry.shape
        out = np.array([[]])
        self.reset_data()
        for i in range(p):
            vec = entry[:, i].reshape(-1, 1)
            out = super().compute(vec, memorize)
        return out

    def descend_gradient(self: Recurrent, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        _, p = self.input.shape
        new_gradient = None
        for i in reversed(range(p)):
            gradient = super().descend_gradient(gradient)
            if new_gradient is None:
                new_gradient = gradient
            else:
                new_gradient = np.vstack((gradient, new_gradient))
        if new_gradient is None:
            raise ValueError
        return new_gradient
