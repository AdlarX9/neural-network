from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class Scale(Layer):
    def __init__(self: Scale, factor: float = 1, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)
        self.factor: float = factor

    def feed_forward(self: Scale, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.factor * entry

    def descend_gradient(self: Scale, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        return self.factor * gradient

    def get_data(self: Scale) -> dict:
        data = super().get_data()
        data["factor"] = self.factor
        return data

    def load_from_data(self: Scale, data: dict) -> None:
        super().load_from_data(data)
        self.factor = data["factor"]
