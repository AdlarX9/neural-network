from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
import math


def RoPE(x: NDArray[np.float64]) -> NDArray[np.float64]:
    _, n, p = x.shape
    if n % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {n}")
    rotated = np.empty_like(x)
    rotated[:, 0::2, :] = -x[:, 1::2, :]
    rotated[:, 1::2, :] = x[:, 0::2, :]
    positions = np.arange(p)[None, :]
    omega = 10000 ** (-2 * (np.arange(n) // 2) / n)[:, None]
    theta = omega @ positions
    theta = theta[None, :, :]
    return x * np.cos(theta) + rotated * np.sin(theta)


def softmax(X: NDArray[np.float64]) -> NDArray[np.float64]:
    X_max = np.max(X, axis=1, keepdims=True)
    X_exp = np.exp(X - X_max)
    Y = X_exp / np.sum(X_exp, axis=1, keepdims=True)
    return Y


class MHA(Layer):
    def __init__(self: MHA, H: int = 1) -> None:
        super().__init__()
        self.H = H
        self.Wq = np.array([[[]]])
        self.Wk = np.array([[[]]])
        self.Wv = np.array([[[]]])
        self.Wo = np.array([[]])
        self.d_h: int = 0

        # Cache
        self.concat = np.array([[]])
        self.V = np.array([[[]]])

    def set_input_shape(self: MHA, input_shape: tuple) -> tuple:
        super().set_input_shape(input_shape)
        n, p = input_shape
        if int(n / self.H) != n / self.H:
            raise ValueError("Dimensions mismatch")
        self.d_h = n / self.H
        self.Wq = np.random.normal(0, np.sqrt(2 / n), size=(self.H, n, p))  # He
        self.Wk = np.random.normal(0, np.sqrt(2 / n), size=(self.H, n, p))  # He
        self.Wv = np.random.normal(0, np.sqrt(2 / n), size=(self.H, n, p))  # He
        self.Wo = np.random.normal(0, np.sqrt(2 / n), size=(n, p))  # He
        return self.output_shape

    def feed_forward(self: MHA, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        d, T = entry.shape

        x = np.expand_dims(entry, axis=0)
        Q = self.Wq @ x
        K = self.Wk @ x
        self.V = self.Wv @ x
        Q = RoPE(Q)
        K = RoPE(K)
        Q = Q.swapaxes(1, 2)
        S = Q @ K / math.sqrt(self.d_h)
        mask = np.triu(np.full(S.shape, -np.inf), k=0)
        S += mask
        A = softmax(S)
        O = self.V @ A

        self.concat = O.reshape(self.H * d, T)
        out = self.Wo @ self.concat
        return out

    def descend_gradient(self: MHA, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        d, T = self.input.shape

        d_concat = self.Wo.T @ gradient
        self.Wo -= self.lr * gradient @ self.concat.T
        d_O = d_concat.reshape(self.H, d, T)
        d_V = d_O @ self.V.T
        # TODO: Finish gradient backpropagation
        return super().descend_gradient(gradient)
