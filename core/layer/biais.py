from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class Biais(Layer):
    def __init__(self: Biais, receive: int = 0) -> None:
        super().__init__((receive,))
        self.B = np.array([[]])

    def set_input_shape(self: Biais, input_shape: tuple[tuple[int, int]]) -> tuple[tuple[int, int]]:
        super().set_input_shape(input_shape)
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0][0]), size=(input_shape[0][0], 1))  # He
        return input_shape

    def feed_forward(self: Biais, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(entry.shape) == 2:
            return entry + self.B
        elif len(entry.shape) == 3:
            return entry + np.expand_dims(self.B, axis=2)
        else:
            raise ValueError

    def descend_gradient(self: Biais, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        if len(self.input[0].shape) == 2:
            self.B -= self.lr * np.sum(gradient, axis=1, keepdims=True)
        elif len(self.input[0].shape) == 3:
            for i in range(self.B.shape[0]):
                self.B[i, 0] -= self.lr * np.sum(gradient[i, :, :])
        else:
            raise ValueError
        return gradient

    def get_data(self: Biais) -> dict:
        data = super().get_data()
        data["B"] = self.B.flatten().tolist()
        return data

    def load_from_data(self: Biais, data: dict) -> None:
        super().load_from_data(data)
        self.B = np.array(data["B"]).reshape(self.input_shape[0][0], 1)
