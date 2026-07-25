from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class UpSample(Layer):
    def __init__(self: UpSample, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def feed_forward(self: UpSample, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.repeat(np.repeat(entry, self.factor, axis=1), self.factor, axis=2)

    def descend_gradient(self: UpSample, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        C, Hout, Wout = gradient.shape
        H = Hout // self.factor
        W = Wout // self.factor
        return gradient.reshape(C, H, self.factor, W, self.factor).sum(axis=(2, 4))

    def get_data(self: UpSample) -> tuple[list[int], list[float], list[str]]:
        int_list, float_list, str_list = super().get_data()
        int_list.append(self.factor)
        return int_list, float_list, str_list

    def load_from_data(
        self: UpSample, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.factor = int_list.pop()
        return super().load_from_data(int_list, float_list, string_list)
