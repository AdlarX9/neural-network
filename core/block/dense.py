from __future__ import annotations
from typing import Any
from .block import Block
from core.layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Dense(Block):
    def __init__(self: Dense, *layers: Layer):
        super().__init__(*layers)

    def set_input_shape(self: Dense, input_shape: tuple) -> tuple:
        block_output_shape = super().set_input_shape(input_shape)
        self.output_shape = (input_shape[0] + block_output_shape[0],) + input_shape[1:]
        return self.output_shape

    def compute(self: Dense, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        if memorize:
            self.input = entry
        output = super().compute(entry, memorize)
        return np.concatenate((entry, output), axis=0)

    def backprop(self: Dense, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        C = self.input.shape[0]
        gradient_x, gradient_block = gradient[:C], gradient[C:]
        gradient_block = super().backprop(gradient_block)
        return gradient_x + gradient_block
