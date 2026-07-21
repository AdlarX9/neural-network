from __future__ import annotations
from .layer import Layer
import numpy as np
from numpy.typing import NDArray


class BN(Layer):
    def __init__(self: BN):
        super().__init__()
        self.gamma: NDArray[np.float64] = np.array([[]])
        self.beta: NDArray[np.float64] = np.array([[]])
        self.epsilon = 1e-5
        self.mean = np.var([[]])
        self.var = np.var([[]])
        self.x_hat = np.var([[]])

    def set_input_shape(self: BN, input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        C, _, _ = input_shape
        self.gamma = np.ones(C)
        self.beta = np.zeros(C)
        return input_shape

    def feed_forward(self: BN, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        C, H, W = entry.shape
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

        self.gamma -= self.lr * np.sum(gradient * self.x_hat, axis=(1, 2))
        self.beta -= self.lr * np.sum(gradient, axis=(1, 2))

        dx_hat = gradient * self.gamma[:, None, None]
        sum_dxhat = np.sum(dx_hat, axis=(1, 2), keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=(1, 2), keepdims=True)
        dx = 1 / np.sqrt(self.var + self.epsilon) * (dx_hat - sum_dxhat / m - self.x_hat * sum_dxhat_xhat / m)
        return dx
