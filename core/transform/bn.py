from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class BN(Layer):
    def __init__(self: BN, receive: tuple[int] = (0,)):
        super().__init__(receive)
        self.gamma: NDArray[np.float64] = np.array([[]])
        self.beta: NDArray[np.float64] = np.array([[]])
        self.epsilon = 1e-5
        self.mean = np.var([[]])
        self.var = np.var([[]])
        self.x_hat = np.var([[]])

    def set_input_shape(self: BN, input_shape: tuple[tuple[int, int, int]]) -> tuple[tuple[int, int, int]]:
        C, _, _ = input_shape[0]
        self.gamma = np.ones((C, 1))
        self.beta = np.zeros((C, 1))
        super().set_input_shape(input_shape)
        return self.output_shape

    def feed_forward(self: BN, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        # moyenne par canal
        self.mean = np.mean(entry, axis=(1, 2), keepdims=True)
        # variance par canal
        self.var = np.var(entry, axis=(1, 2), keepdims=True)
        self.x_hat = (entry - self.mean) / np.sqrt(self.var + self.epsilon)
        output = self.gamma[:, None, None] * self.x_hat + self.beta[:, None, None]
        return output

    def descend_gradient(self: BN, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None or self.x_hat is None or self.gamma is None or self.beta is None:
            raise MemoryError
        _, H, W = gradient.shape
        m = H * W

        dx_hat = gradient * self.gamma[:, None, None]
        sum_dxhat = np.sum(dx_hat, axis=(1, 2), keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=(1, 2), keepdims=True)
        dx = 1 / np.sqrt(self.var + self.epsilon) * (dx_hat - sum_dxhat / m - self.x_hat * sum_dxhat_xhat / m)

        self.gamma -= self.lr * np.sum(gradient * self.x_hat, axis=(1, 2))
        self.beta -= self.lr * np.sum(gradient, axis=(1, 2))
        return dx

    def get_data(self: BN) -> dict:
        data = super().get_data()
        data["gamma"] = self.gamma.flatten().tolist()
        data["beta"] = self.beta.flatten().tolist()
        return data

    def load_from_data(self: BN, data: dict) -> None:
        super().load_from_data(data)
        self.gamma = np.array(data["gamma"]).reshape(self.input_shape[0][0], 1)
        self.beta = np.array(data["beta"]).reshape(self.input_shape[0][0], 1)
