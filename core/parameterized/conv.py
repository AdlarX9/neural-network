from __future__ import annotations
from typing import Any
import numpy as np
from ..basics.layer import Layer
from ..utils.typing import Shape, ShapeFlow, Tensor, ParamGrad, SaveData


def im2col(x: Tensor, K: int, S: int, P: int) -> Tensor:
    C, H, W = x.shape
    # Padding spatial
    x_padded = np.pad(x, ((0, 0), (P, P), (P, P)), mode="constant", constant_values=0)
    _, H_padded, W_padded = x_padded.shape
    Hout = (H_padded - K) // S + 1
    Wout = (W_padded - K) // S + 1
    cols = np.empty((C * K * K, Hout * Wout))
    col = 0
    for i in range(Hout):
        for j in range(Wout):
            patch = x_padded[
                :,
                i * S : i * S + K,
                j * S : j * S + K,
            ]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols


def col2im(cols: Tensor, input_shape: Shape, K: int, S: int, P: int) -> Tensor:
    C, H, W = input_shape
    H_padded = H + 2 * P
    W_padded = W + 2 * P
    Hout = (H_padded - K) // S + 1
    Wout = (W_padded - K) // S + 1
    out_padded = np.zeros((C, H_padded, W_padded))
    col = 0
    for i in range(Hout):
        for j in range(Wout):
            patch = cols[:, col]
            patch = patch.reshape(C, K, K)
            out_padded[
                :,
                i * S : i * S + K,
                j * S : j * S + K,
            ] += patch
            col += 1
    # Suppression du padding
    if P == 0:
        return out_padded
    return out_padded[:, P:-P, P:-P]


class Conv(Layer):
    def __init__(self: Conv, N: int = 0, K: int = 0, S: int = 0, P: int = 0, receive: int = 0):
        super().__init__((receive,))
        self.K = K  # Watching field dimension
        self.S = S  # Stride
        self.N = N
        self.Xcol: Tensor | None = None
        self.kernels: Tensor = np.array([[[[]]]])
        self.parameters = ["kernels"]
        self.kernels_grad: Tensor | None = None
        self.P = P
        if self.P == -1:
            if self.K % 2 == 1:
                self.P = (self.K - 1) // 2
            else:
                raise ValueError("K must not be even:", self.K)

    def set_input_shape(self: Conv, input_shape: ShapeFlow) -> ShapeFlow:
        c, H, W = input_shape[0]
        super().set_input_shape(input_shape)
        self.kernels = np.random.normal(0, np.sqrt(2 / (c * self.K**2)), size=(self.N, c, self.K, self.K))
        Hout = (H + 2 * self.P - self.K) // self.S + 1
        Wout = (W + 2 * self.P - self.K) // self.S + 1
        self.output_shape = ((self.N, Hout, Wout),)
        return self.output_shape

    def feed_forward(self: Conv, entry: Tensor) -> Tensor:
        Xcol = im2col(entry, self.K, self.S, self.P)
        self.Xcol = Xcol
        W = self.kernels.reshape(self.N, -1)
        Y = W @ Xcol
        _, H, Winput = entry.shape
        Hout = (H + 2 * self.P - self.K) // self.S + 1
        Wout = (Winput + 2 * self.P - self.K) // self.S + 1
        return Y.reshape(self.N, Hout, Wout)

    def descend_gradient(self: Conv, gradient: Tensor) -> Tensor:
        if self.input is None or self.Xcol is None:
            raise MemoryError
        delta = gradient.reshape(self.N, -1)
        Wgrad = delta @ self.Xcol.T
        self.kernels_grad = Wgrad.reshape(self.kernels.shape)
        dXcol = self.kernels.reshape(self.N, -1).T @ delta
        return col2im(dXcol, self.input[0].shape, self.K, self.S, self.P)

    def params_gradient(self: Conv, gradient) -> ParamGrad:
        if self.kernels_grad is None:
            raise MemoryError
        return {"kernels": self.kernels_grad}

    def get_data(self: Conv) -> SaveData:
        data = super().get_data()
        data["K"] = self.K
        data["S"] = self.S
        data["N"] = self.N
        data["P"] = self.P
        return data

    def load_from_data(self: Conv, data: SaveData) -> None:
        super().load_from_data(data)
        self.K = data["K"]
        self.S = data["S"]
        self.N = data["N"]
        self.P = data["P"]
