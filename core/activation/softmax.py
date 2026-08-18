from __future__ import annotations
from typing import Any
from .activation import Activation
import numpy as np
from numpy.typing import NDArray


class Softmax(Activation):
    def __init__(self: Softmax, axis: int | None = None, receive: tuple[int] = (0,)) -> None:
        self.axis = axis
        self.output: NDArray[np.float64] | None = None
        super().__init__(receive)

    def feed_forward(self: Softmax, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        max_entry = np.max(entry, axis=self.axis, keepdims=True)
        exp_entry = np.exp(entry - max_entry)
        softmax: NDArray[np.float64] = exp_entry / np.sum(exp_entry, axis=self.axis, keepdims=True)
        self.output = softmax
        return self.output

    def descend_gradient(self: Softmax, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.output is None:
            raise MemoryError
        return self.output * (gradient - np.sum(gradient * self.output, axis=self.axis, keepdims=True))

    def get_data(self: Softmax) -> dict:
        data = super().get_data()
        data["axis"] = self.axis
        return data

    def load_from_data(self: Softmax, data: dict) -> None:
        super().load_from_data(data)
        self.axis = data["axis"]
