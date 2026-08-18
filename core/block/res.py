from __future__ import annotations
from .adder import Adder
from ..layer.layer import Layer


class Res(Adder):
    def __init__(self: Res, layer: Layer = Layer(), receive: tuple[int] = (0,)) -> None:
        super().__init__([layer, Layer()], receive)  # Add Identity

    def set_input_shape(self: Res, input_shape: tuple[tuple]) -> tuple[tuple]:
        super().set_input_shape(input_shape)
        if self.layers[0].input_shape != self.layers[0].output_shape:
            raise ValueError(
                "ResBlock modifies data dimension:",
                self.layers[0].input_shape,
                "=>",
                self.layers[0].output_shape,
            )
        return self.output_shape
