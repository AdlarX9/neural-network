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

    def get_data(self: RMSNorm) -> tuple[list[int], list[float], list[str]]:
        int_list, float_list, str_list = super().get_data()
        float_list += self.gamma.flatten().tolist()
        return int_list, float_list, str_list

    def load_from_data(
        self: RMSNorm, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.input_shape = tuple(int_list[:2])
        del int_list[:2]
        self.lr = float_list.pop(0)

        self.gamma = np.array(float_list[: self.input_shape[0]]).reshape((self.input_shape[0], 1))
        del float_list[: self.input_shape[0]]
