from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
import math
from ..utils.functions import softmax, RoPE


class MHA(Layer):
    def __init__(self: MHA, H: int = 1, causal: bool = False) -> None:
        super().__init__()
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

    def set_input_shape(self: MHA, input_shape: tuple) -> tuple:
        super().set_input_shape(input_shape)
        d, _ = input_shape
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
        _, T = self.input.shape

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
        self.Wq -= self.lr * d_Q @ self.input.T
        self.Wk -= self.lr * d_K @ self.input.T
        self.Wv -= self.lr * d_V @ self.input.T
        self.Wo -= self.lr * gradient @ self.concat.T
        return d_X

    def get_data(self: MHA) -> tuple[list[int], list[float], list[str]]:
        int_list, float_list, str_list = super().get_data()
        int_list += [self.H, self.d_h, int(self.causal)]
        float_list += self.Wq.flatten().tolist()
        float_list += self.Wk.flatten().tolist()
        float_list += self.Wv.flatten().tolist()
        float_list += self.Wo.flatten().tolist()
        return int_list, float_list, str_list

    def load_from_data(
        self: MHA, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.input_shape = tuple(int_list[:2])
        del int_list[:2]
        self.lr = float_list.pop(0)
        self.H = int_list.pop(0)
        self.d_h = int_list.pop(0)
        self.causal = bool(int_list.pop(0))
        d = self.input_shape[0]

        self.Wq = np.array(float_list[: self.H * self.d_h * d]).reshape((self.H, self.d_h, d))
        del float_list[: self.H * self.d_h * d]
        self.Wk = np.array(float_list[: self.H * self.d_h * d]).reshape((self.H, self.d_h, d))
        del float_list[: self.H * self.d_h * d]
        self.Wv = np.array(float_list[: self.H * self.d_h * d]).reshape((self.H, self.d_h, d))
        del float_list[: self.H * self.d_h * d]
        self.Wo = np.array(float_list[: d**2]).reshape((d, d))
        del float_list[: d**2]
