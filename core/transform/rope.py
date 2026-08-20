from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from ..utils.typing import Tensor, Receive1


class RoPE(Layer):
    def __init__(self: RoPE, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

    def _rope(self: RoPE, x: Tensor, factor: int = 1) -> Tensor:
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

    def feed_forward(self: RoPE, entry: Tensor) -> Tensor:
        return self._rope(x=entry, factor=1)

    def descend_gradient(self: RoPE, gradient: Tensor) -> Tensor:
        return self._rope(x=gradient, factor=-1)
