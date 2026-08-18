from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class FC(Layer):
    def __init__(self: FC, n: int = 0, receive: int = 0) -> None:
        super().__init__((receive,))
        self.n: int = n
        self.p: int = 0
        self.W = np.array([[]])

    def set_input_shape(self: FC, input_shape: tuple[tuple[int, int]]) -> tuple[tuple[int, int]]:
        super().set_input_shape(input_shape)
        p, q = input_shape[0]
        self.p = p
        self.W = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))  # He
        self.output_shape = ((self.n, q),)
        return self.output_shape

    def feed_forward(self: FC, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.W @ entry

    def descend_gradient(self: FC, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = self.W.T @ gradient
        self.W -= self.lr * (gradient @ self.input[0].T)
        return new_gradient

    def get_data(self: FC) -> dict:
        data = super().get_data()
        data["n"] = self.n
        data["p"] = self.p
        data["W"] = self.W.flatten().tolist()
        return data

    def load_from_data(self: FC, data: dict) -> None:
        super().load_from_data(data)
        self.n = data["n"]
        self.p = data["p"]
        self.W = np.array(data["W"]).reshape((self.n, self.p))
