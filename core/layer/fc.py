from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class FC(Layer):
    def __init__(self: FC, n: int = 0) -> None:
        super().__init__()
        self.n: int = n
        self.p: int = 0
        self.W = np.array([[]])

    def set_input_shape(self: FC, input_shape: tuple[int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        p, q = input_shape
        self.p = p
        self.W = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))  # He
        self.output_shape = (self.n, q)
        return self.output_shape

    def feed_forward(self: FC, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.W @ entry

    def descend_gradient(self: FC, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = self.W.T @ gradient
        self.W -= self.lr * (gradient @ self.input.T)
        return new_gradient

    def get_data(self: FC) -> tuple[list[int], list[float], list[str]]:
        int_list = list(self.input_shape) + [self.n, self.p]
        float_list = [self.lr] + self.W.flatten().tolist()
        return int_list, float_list, []

    def load_from_data(self: FC, int_list: list[int], float_list: list[float], string_list: list[str]) -> None:
        self.input_shape = tuple(int_list[:2])
        self.n = int_list[2]
        self.p = int_list[3]
        self.lr = float_list.pop(0)
        self.W = np.array(float_list).reshape(self.n, self.p)
