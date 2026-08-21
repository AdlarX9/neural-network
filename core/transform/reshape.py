from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..utils.typing import Shape, ShapeFlow, Tensor, Receive1, SaveData
from ..basics.layer import Layer


class Reshape(Layer):
    def __init__(self: Reshape, shape: Shape = (-1, 1), receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.shape: Shape = shape

    def set_input_shape(self: Reshape, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        self.output_shape = (self.shape,)
        return self.output_shape

    def feed_forward(self: Reshape, entry: Tensor) -> Tensor:
        return entry.reshape(self.shape)

    def descend_gradient(self: Reshape, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        return gradient.reshape(self.input[0].shape)

    def get_data(self: Reshape) -> SaveData:
        data = super().get_data()
        data["shape"] = self.shape
        return data

    def load_from_data(self: Reshape, data: SaveData) -> None:
        super().load_from_data(data)
        self.shape = data["shape"]
