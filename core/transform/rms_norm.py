from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class RMSNorm(Layer):
    def __init__(self: RMSNorm) -> None:
        super().__init__()
        self.gamma: NDArray[np.float64] = np.array([[]])
        self.rms: NDArray[np.float64] = np.array([[]])

    def set_input_shape(self: RMSNorm, input_shape: tuple) -> tuple:
        self.gamma = np.ones((input_shape[0], 1))
        return super().set_input_shape(input_shape)

    def feed_forward(self: RMSNorm, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        self.rms = np.sqrt(np.sum(entry**2, axis=0) / entry.shape[0] + 1e-12).reshape(1, -1)
        return self.gamma * entry / self.rms

    def descend_gradient(self: RMSNorm, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        d = self.input.shape[0]
        dot = np.sum(gradient * self.input, axis=0, keepdims=True)
        new_gradient = self.gamma / self.rms * (gradient - self.input * dot / (d * self.rms**2))

        self.gamma -= self.lr * np.sum(gradient * self.input / self.rms, axis=1, keepdims=True)
        return new_gradient

    def get_data(self: RMSNorm) -> dict:
        data = super().get_data()
        data["gamma"] = self.gamma.flatten().tolist()
        return data

    def load_from_data(self: RMSNorm, data: dict) -> None:
        super().load_from_data(data)
        self.gamma = np.array(data["gamma"]).reshape(self.input_shape[0], 1)
