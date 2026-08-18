from __future__ import annotations
from .block import Block
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Adder(Block):
    def __init__(self: Adder, layers: list[Layer] = [], receive: tuple[int] = (0,)) -> None:
        super().__init__(layers, receive)

    def set_input_shape(self: Adder, input_shape: tuple[tuple]) -> tuple[tuple]:
        self.input_shape = input_shape
        if len(self.layers) == 0:
            return input_shape
        for layer in self.layers:
            layer.set_input_shape(input_shape)
        self.output_shape = self.layers[0].output_shape
        for layer in self.layers:
            if self.output_shape != layer.output_shape:
                raise ValueError(
                    "Adder data dimension incorrect:",
                    "\n".join([str(layer.output_shape) for layer in self.layers]),
                )
        return self.output_shape

    def __call__(
        self: Adder, entry: tuple[NDArray[np.float64]], memorize: bool
    ) -> tuple[NDArray[np.float64]]:
        output: NDArray[np.float64] | None = None
        if memorize:
            self.input = entry
        for layer in self.layers:
            if output is None:
                output = layer(entry, memorize)[0]
            else:
                output += layer(entry, memorize)[0]
        if output is None:
            return (np.zeros(self.output_shape[0]),)
        return (output,)

    def backprop(self: Adder, gradient: tuple[NDArray[np.float64]]) -> tuple[NDArray[np.float64]]:
        if self.input is None:
            raise MemoryError
        new_gradient = np.zeros_like(self.input[0])
        for layer in self.layers:
            new_gradient += layer.backprop(gradient)[0]
        return (new_gradient,)
