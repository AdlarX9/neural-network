from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class Reshape(Layer):
    def __init__(self: Reshape, shape: tuple[int, ...] = (-1, 1), receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)
        self.shape: tuple[int, ...] = shape

    def set_input_shape(self: Reshape, input_shape: tuple[tuple[int, int, int]]) -> tuple[tuple[int, int]]:
        super().set_input_shape(input_shape)
        self.output_shape = (self.shape,)
        return self.output_shape

    def feed_forward(self: Reshape, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry.reshape(self.shape)

    def descend_gradient(self: Reshape, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        return gradient.reshape(self.input[0].shape)

    def get_data(self: Reshape) -> dict:
        data = super().get_data()
        data["shape"] = self.shape
        return data

    def load_from_data(self: Reshape, data: dict) -> None:
        super().load_from_data(data)
        self.shape = data["shape"]
