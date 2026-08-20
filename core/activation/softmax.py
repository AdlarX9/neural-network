from __future__ import annotations
from .activation import Activation
import numpy as np
from ..utils.typing import Tensor, Receive1, SaveData


class Softmax(Activation):
    def __init__(self: Softmax, axis: int | None = None, receive: Receive1 = (0,)) -> None:
        self.axis = axis
        self.output: Tensor | None = None
        super().__init__(receive)

    def feed_forward(self: Softmax, entry: Tensor) -> Tensor:
        max_entry = np.max(entry, axis=self.axis, keepdims=True)
        exp_entry = np.exp(entry - max_entry)
        softmax: Tensor = exp_entry / np.sum(exp_entry, axis=self.axis, keepdims=True)
        self.output = softmax
        return self.output

    def descend_gradient(self: Softmax, gradient: Tensor) -> Tensor:
        if self.output is None:
            raise MemoryError
        return self.output * (gradient - np.sum(gradient * self.output, axis=self.axis, keepdims=True))

    def get_data(self: Softmax) -> SaveData:
        data = super().get_data()
        data["axis"] = self.axis
        return data

    def load_from_data(self: Softmax, data: SaveData) -> None:
        super().load_from_data(data)
        self.axis = data["axis"]
