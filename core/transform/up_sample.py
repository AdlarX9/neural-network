from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class UpSample(Layer):
    def __init__(self: UpSample, factor: int = 2, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)
        self.factor = factor

    def feed_forward(self: UpSample, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.repeat(np.repeat(entry, self.factor, axis=1), self.factor, axis=2)

    def descend_gradient(self: UpSample, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        C, Hout, Wout = gradient.shape
        H = Hout // self.factor
        W = Wout // self.factor
        return gradient.reshape(C, H, self.factor, W, self.factor).sum(axis=(2, 4))

    def get_data(self: UpSample) -> dict:
        data = super().get_data()
        data["factor"] = self.factor
        return data

    def load_from_data(self: UpSample, data: dict) -> None:
        super().load_from_data(data)
        self.factor = data["factor"]
