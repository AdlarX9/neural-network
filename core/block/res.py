from __future__ import annotations
from .block import Block
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Res(Block):
    def __init__(self: Res, *layers: Layer) -> None:
        super().__init__(*layers)

    def set_input_shape(self: Res, input_shape: tuple) -> tuple:
        super().set_input_shape(input_shape)
        if self.input_shape != self.output_shape:
            raise ValueError("ResBlock modifies data dimension :", self.input_shape, "=>", self.output_shape)
        return self.output_shape

    def compute(self: Res, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        correction = super().compute(entry, memorize)
        return entry + correction

    def backprop(self: Res, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        new_gradient = super().backprop(gradient)
        return new_gradient + gradient
