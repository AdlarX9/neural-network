from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class ConvBiais(Layer):
    def __init__(self: ConvBiais, receive: int = 0) -> None:
        super().__init__((receive,))
        self.B = np.array([[]])

    def set_input_shape(self: ConvBiais, input_shape: tuple[tuple[int, int, int]]) -> tuple[tuple[int, int, int]]:
        super().set_input_shape(input_shape)
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0][0]), size=(input_shape[0][0], 1))  # He
        return input_shape

    def feed_forward(self: ConvBiais, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry + np.expand_dims(self.B, axis=2)

    def descend_gradient(self: ConvBiais, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        for i in range(self.B.shape[0]):
            self.B[i, 0] -= self.lr * np.sum(gradient[i, :, :])
        return gradient

    def get_data(self: ConvBiais) -> dict:
        data = super().get_data()
        data["B"] = self.B.flatten().tolist()
        return data

    def load_from_data(self: ConvBiais, data: dict) -> None:
        super().load_from_data(data)
        self.B = np.array(data["B"]).reshape(self.input_shape[0][0], 1)
