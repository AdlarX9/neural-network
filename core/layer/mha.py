from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
import math
from ..utils.functions import softmax, RoPE


class MHA(Layer):
    def __init__(self: MHA, H: int = 1, causal: bool = False, receive: int = 0) -> None:
        super().__init__((receive,))
        self.H = H
        self.Wq = np.array([[[]]])
        self.Wk = np.array([[[]]])
        self.Wv = np.array([[[]]])
        self.Wo = np.array([[]])
        self.d_h: int = 0
        self.causal = causal

        # Cache
        self.concat = np.array([[]])
        self.Q = np.array([[[]]])
        self.K = np.array([[[]]])
        self.V = np.array([[[]]])
        self.A = np.array([[[]]])
        self.S = np.array([[[]]])

    def set_input_shape(self: MHA, input_shape: tuple[tuple[int, int]]) -> tuple[tuple[int, int]]:
        super().set_input_shape(input_shape)
        d, _ = input_shape[0]
        if int(d / self.H) != d / self.H:
            raise ValueError("Dimensions mismatch")
        self.d_h = d // self.H
        self.Wq = np.random.normal(0, np.sqrt(2 / d), size=(self.H, self.d_h, d))  # He
        self.Wk = np.random.normal(0, np.sqrt(2 / d), size=(self.H, self.d_h, d))  # He
        self.Wv = np.random.normal(0, np.sqrt(2 / d), size=(self.H, self.d_h, d))  # He
        self.Wo = np.random.normal(0, np.sqrt(2 / d), size=(d, d))  # He
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
        if self.causal:
            mask = np.tril(np.full(S.shape, -np.inf), k=-1)
            S += mask
        self.A = softmax(S, axis=1)
        O = self.V @ self.A

        self.concat = O.reshape(self.H * self.d_h, T)
        out = self.Wo @ self.concat
        return out

    def descend_gradient(self: MHA, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        _, T = self.input[0].shape

        # Compute gradients
        d_concat = self.Wo.T @ gradient
        d_O = d_concat.reshape(self.H, self.d_h, T)
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
        transposed = self.input[0].T
        self.Wq -= self.lr * d_Q @ transposed
        self.Wk -= self.lr * d_K @ transposed
        self.Wv -= self.lr * d_V @ transposed
        self.Wo -= self.lr * gradient @ self.concat.T
        return d_X

    def get_data(self: MHA) -> dict:
        data = super().get_data()
        data["H"] = self.H
        data["d_h"] = self.d_h
        data["causal"] = self.causal

        data["Wq"] = self.Wq.flatten().tolist()
        data["Wk"] = self.Wk.flatten().tolist()
        data["Wv"] = self.Wv.flatten().tolist()
        data["Wo"] = self.Wo.flatten().tolist()

        return data

    def load_from_data(self: MHA, data: dict) -> None:
        super().load_from_data(data)
        self.H = data["H"]
        self.d_h = data["d_h"]
        self.causal = data["causal"]

        d = self.input_shape[0][0]
        self.Wq = np.array(data["Wq"]).reshape(self.H, self.d_h, d)
        self.Wk = np.array(data["Wk"]).reshape(self.H, self.d_h, d)
        self.Wv = np.array(data["Wv"]).reshape(self.H, self.d_h, d)
        self.Wo = np.array(data["Wo"]).reshape(d, d)
