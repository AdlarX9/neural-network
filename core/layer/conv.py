from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


def im2col(x: NDArray[np.float64], K: int, S: int) -> NDArray[np.float64]:
    C, H, W = x.shape
    Hout = (H - K) // S + 1
    Wout = (W - K) // S + 1
    cols = np.empty((C * K * K, Hout * Wout))
    col = 0
    for i in range(Hout):
        for j in range(Wout):
            patch = x[:, i * S : i * S + K, j * S : j * S + K]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols


def col2im(
    cols: NDArray[np.float64], input_shape: tuple[int, int, int], K: int, S: int
) -> NDArray[np.float64]:
    C, H, W = input_shape
    Hout = (H - K) // S + 1
    Wout = (W - K) // S + 1
    out = np.zeros(input_shape)
    col = 0
    for i in range(Hout):
        for j in range(Wout):
            patch = cols[:, col]
            patch = patch.reshape(C, K, K)
            out[:, i * S : i * S + K, j * S : j * S + K] += patch
            col += 1
    return out


class Conv(Layer):
    def __init__(self: Conv, N: int = 0, K: int = 0, S: int = 0):
        super().__init__()
        self.K = K  # Watching field dimension
        self.S = S  # Stride
        self.N = N
        self.Xcol: NDArray[np.float64] | None = None
        self.kernels: NDArray[np.float64] = np.array([[[[]]]])

    def set_input_shape(self: Conv, input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        c, n, p = input_shape
        self.input_shape = input_shape
        self.kernels = np.random.normal(
            0, np.sqrt(2 / (c * self.K**2)), size=(self.N, c, self.K, self.K)
        )  # He
        n_o = (n - self.K) // self.S + 1
        p_o = (p - self.K) // self.S + 1
        return (self.N, n_o, p_o)

    def feed_forward(self: Conv, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input = entry
        Xcol = im2col(entry, self.K, self.S)
        self.Xcol = Xcol
        W = self.kernels.reshape(self.N, -1)
        Y = W @ Xcol
        _, H, Winput = entry.shape
        Hout = (H - self.K) // self.S + 1
        Wout = (Winput - self.K) // self.S + 1
        return Y.reshape(self.N, Hout, Wout)

    def descend_gradient(self: Conv, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None or self.Xcol is None:
            raise MemoryError
        delta = gradient.reshape(self.N, -1)
        Wgrad = delta @ self.Xcol.T
        Wgrad = Wgrad.reshape(self.kernels.shape)
        dXcol = self.kernels.reshape(self.N, -1).T @ delta
        self.kernels -= self.lr * Wgrad
        return col2im(dXcol, self.input.shape, self.K, self.S)

    def get_data(self: Conv) -> tuple[list[int], list[float], list[str]]:
        int_list = list(self.input_shape) + [self.K, self.S, self.N]
        float_list = [self.lr]
        float_list += self.kernels.flatten().tolist()
        return int_list, float_list, []

    def load_from_data(
        self: Conv, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.input_shape = tuple(int_list[:3])
        self.K = int_list[3]
        self.S = int_list[4]
        self.N = int_list[5]
        self.lr = float_list[0]
        del float_list[0]
        self.kernels = np.array(float_list).reshape(self.N, self.input_shape[0], self.K, self.K)
