from __future__ import annotations
from .block import Block
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Adder(Block):
    def __init__(self: Adder, *layers: Layer) -> None:
        super().__init__(*layers)

    def set_input_shape(self: Adder, input_shape: tuple) -> tuple:
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

    def compute(self: Adder, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        output = np.zeros(self.output_shape)
        for layer in self.layers:
            output += layer.compute(entry, memorize)
        return output

    def backprop(self: Adder, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        new_gradient = np.zeros(self.input_shape)
        for layer in self.layers:
            new_gradient += layer.backprop(gradient)
        return new_gradient
