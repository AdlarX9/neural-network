from __future__ import annotations
from .block import Block
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Multiplier(Block):
    def __init__(self: Multiplier, *layers: Layer) -> None:
        super().__init__(*layers)
        self.outputs: list[NDArray[np.float64]] = []

    def set_input_shape(self: Multiplier, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        if len(self.layers) == 0:
            return input_shape
        for layer in self.layers:
            layer.set_input_shape(input_shape)
        self.output_shape = self.layers[0].input_shape
        for layer in self.layers:
            if layer.input_shape != layer.output_shape:
                raise ValueError(
                    "Multiplier modifies data dimension:", layer.input_shape, "=>", layer.output_shape
                )
        return self.output_shape

    def compute(self: Multiplier, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        output = np.ones(self.output_shape)
        self.outputs = []
        for layer in self.layers:
            result = layer.compute(entry, memorize)
            if memorize:
                self.outputs.append(result)
            output *= result
        return output

    def backprop(self: Multiplier, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = np.zeros(self.input.shape)
        for i in range(len(self.layers)):
            term = np.ones(self.input.shape)
            for j in range(i):
                if i != j:
                    term *= self.outputs[j]
            term *= self.layers[i].backprop(gradient)
            new_gradient += term
        return new_gradient
