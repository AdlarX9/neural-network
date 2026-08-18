from __future__ import annotations
from .block import Block
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Multiplier(Block):
    def __init__(self: Multiplier, layers: list[Layer] = [], receive: tuple[int] = (0,)) -> None:
        super().__init__(layers, receive)
        self.outputs: list[tuple[NDArray[np.float64]]] = []

    def set_input_shape(self: Multiplier, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        if len(self.layers) == 0:
            return input_shape
        for layer in self.layers:
            layer.set_input_shape(input_shape)
        self.output_shape = self.layers[0].output_shape
        for layer in self.layers:
            if self.output_shape != layer.output_shape:
                raise ValueError(
                    "Multiplier incompatible data dimension:", self.output_shape, "!=", layer.output_shape
                )
        return self.output_shape

    def __call__(
        self: Multiplier, entry: tuple[NDArray[np.float64]], memorize: bool
    ) -> tuple[NDArray[np.float64]]:
        output = None
        self.outputs = []
        if memorize:
            self.input = entry
        for layer in self.layers:
            result = layer(entry, memorize)
            if memorize:
                self.outputs.append(result)
            if output is None:
                output = result[0]
            else:
                output *= result[0]
        if output is None:
            raise ValueError
        return (output,)

    def backprop(self: Multiplier, gradient: tuple[NDArray[np.float64]]) -> tuple[NDArray[np.float64]]:
        if self.input is None:
            raise MemoryError
        new_gradient = np.zeros_like(self.input[0])
        for i in range(len(self.layers)):
            d_fi = gradient[0].copy()
            for j in range(len(self.layers)):
                if i != j:
                    d_fi *= self.outputs[j][0]
            d_x = self.layers[i].backprop((d_fi,))[0]
            new_gradient += d_x
        return (new_gradient,)
