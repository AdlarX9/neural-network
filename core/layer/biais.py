from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class Biais(Layer):
    def __init__(self: Biais) -> None:
        super().__init__()
        self.B = np.array([[]])

    def set_input_shape(self: Biais, input_shape: tuple[int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0]), size=(input_shape[0], 1))  # He
        return input_shape

    def feed_forward(self: Biais, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry + self.B

    def descend_gradient(self: Biais, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        self.B -= self.lr * np.sum(gradient, axis=1, keepdims=True)
        return gradient

    def get_data(self: Biais) -> tuple[list[int], list[float], list[str]]:
        int_list = list(self.input_shape)
        float_list = [self.lr] + self.B.flatten().tolist()
        return int_list, float_list, []

    def load_from_data(self: Biais, int_list: list[int], float_list: list[float], string_list: list[str]) -> None:
        self.input_shape = tuple(int_list)
        self.lr = float_list.pop(0)
        self.B = np.array(float_list).reshape(self.input_shape[0], 1)
