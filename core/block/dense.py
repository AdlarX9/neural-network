from __future__ import annotations
from .block import Block
from core.layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Dense(Block):
    def __init__(self: Dense, layers: list[Layer] = [], receive: tuple[int] = (0,)):
        super().__init__(layers, receive)

    def set_input_shape(self: Dense, input_shape: tuple[tuple[int, ...]]) -> tuple[tuple[int, ...]]:
        block_output_shape = super().set_input_shape(input_shape)
        self.output_shape = ((input_shape[0][0] + block_output_shape[0][0],) + input_shape[0][1:],)
        return self.output_shape

    def __call__(
        self: Dense, entry: tuple[NDArray[np.float64]], memorize: bool
    ) -> tuple[NDArray[np.float64]]:
        if memorize:
            self.input = entry
        output = super()(entry, memorize)
        return (np.concatenate((entry[0], output[0]), axis=0),)

    def backprop(self: Dense, gradient: tuple[NDArray[np.float64]]) -> tuple[NDArray[np.float64]]:
        if self.input is None:
            raise MemoryError
        C = self.input[0].shape[0]
        gradient_x, gradient_block = gradient[0][:C], gradient[0][C:]
        gradient_block = super().backprop((gradient_block,))[0]
        return (gradient_x + gradient_block,)
