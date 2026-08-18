from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Duplicate(Layer):
    def __init__(self: Duplicate, factor: int = 1, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)
        self.factor: int = factor

    def set_input_shape(self: Duplicate, input_shape: tuple[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
        super().set_input_shape(input_shape)
        self.output_shape = tuple([input_shape[0] for _ in range(self.factor)])
        return self.output_shape

    def feed_forward(self: Duplicate, entry: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        return tuple([entry.copy() for _ in range(self.factor)])

    def descend_gradient(self: Duplicate, gradient: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient: NDArray[np.float64] = np.zeros_like(self.input[0])
        for grad in gradient:
            new_gradient += grad
        return new_gradient

    def get_data(self: Duplicate) -> dict:
        data = super().get_data()
        data["factor"] = self.factor
        return data

    def load_from_data(self: Duplicate, data: dict) -> None:
        super().load_from_data(data)
        self.factor = data["factor"]
