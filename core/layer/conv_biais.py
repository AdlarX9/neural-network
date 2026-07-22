from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class ConvBiais(Layer):
    def __init__(self: ConvBiais) -> None:
        super().__init__()
        self.B = np.array([[]])

    def set_input_shape(self: ConvBiais, input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        self.input_shape = input_shape
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0]), size=(input_shape[0], 1))  # He
        return input_shape

    def feed_forward(self: ConvBiais, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry + np.expand_dims(self.B, axis=2)

    def descend_gradient(self: ConvBiais, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        for i in range(self.B.shape[0]):
            self.B[i, 0] -= self.lr * np.sum(gradient[i, :, :])
        return gradient

    def get_data(self: ConvBiais) -> tuple[list[int], list[float], list[str]]:
        int_list = list(self.input_shape)
        float_list = [self.lr] + self.B.flatten().tolist()
        return int_list, float_list, []

    def load_from_data(self: ConvBiais, int_list: list[int], float_list: list[float], string_list: list[str]) -> None:
        self.input_shape = tuple(int_list)
        self.lr = float_list.pop(0)
        self.B = np.array(float_list).reshape(self.input_shape[0], 1)
