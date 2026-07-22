from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
import math


def RoPE(x: NDArray[np.float64], factor: int = 1) -> NDArray[np.float64]:
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
    return x * np.cos(factor * theta) + rotated * np.sin(factor * theta)


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
        self.Q = np.array([[[]]])
        self.K = np.array([[[]]])
        self.V = np.array([[[]]])
        self.A = np.array([[[]]])
        self.S = np.array([[[]]])

    def set_input_shape(self: MHA, input_shape: tuple) -> tuple:
        super().set_input_shape(input_shape)
        d, T = input_shape
        if int(d / self.H) != d / self.H:
            raise ValueError("Dimensions mismatch")
        self.d_h = d // self.H
        self.Wq = np.random.normal(0, np.sqrt(2 / d), size=(self.H, self.d_h, T))  # He
        self.Wk = np.random.normal(0, np.sqrt(2 / d), size=(self.H, self.d_h, T))  # He
        self.Wv = np.random.normal(0, np.sqrt(2 / d), size=(self.H, self.d_h, T))  # He
        self.Wo = np.random.normal(0, np.sqrt(2 / d), size=(d, T))  # He
        return self.output_shape

    def feed_forward(self: MHA, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        d, T = entry.shape

        x = np.expand_dims(entry, axis=0)
        Q = self.Wq @ x
        K = self.Wk @ x
        self.V = self.Wv @ x
        Q = RoPE(Q)
        self.K = RoPE(K)
        self.Q = Q.swapaxes(1, 2)
        S = self.Q @ self.K / math.sqrt(self.d_h)
        mask = np.tril(np.full(S.shape, -np.inf), k=-1)
        S += mask
        self.A = softmax(S)
        O = self.V @ self.A

        self.concat = O.reshape(self.H * d, T)
        out = self.Wo @ self.concat
        return out

    def descend_gradient(self: MHA, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        d, T = self.input.shape

        # Compute gradients
        d_concat = self.Wo.T @ gradient
        self.Wo -= self.lr * gradient @ self.concat.T
        d_O = d_concat.reshape(self.H, d, T)
        d_V = d_O @ self.A.swapaxes(1, 2)
        d_A = self.V.swapaxes(1, 2) @ d_O
        d_S: NDArray[np.float64] = self.A * (d_A - np.sum(d_A * self.A, axis=1, keepdims=True))
        d_K = self.Q.swapaxes(1, 2) @ d_S / math.sqrt(self.d_h)
        d_Q = d_S @ self.K.swapaxes(1, 2) / math.sqrt(self.d_h)
        d_Q = d_Q.swapaxes(1, 2)
        d_Q = RoPE(d_Q, -1)  # Rotation de -theta annule la rotation de theta
        d_K = RoPE(d_K, -1)  # Rotation de -theta annule la rotation de theta
        d_X: NDArray[np.float64] = np.sum(
            self.Wq.swapaxes(1, 2) @ d_Q + self.Wk.swapaxes(1, 2) @ d_K + self.Wv.swapaxes(1, 2) @ d_V, axis=0
        )

        # Learn weights
        self.Wq -= self.lr * d_Q @ self.input.T
        self.Wk -= self.lr * d_K @ self.input.T
        self.Wv -= self.lr * d_V @ self.input.T
        return d_X
