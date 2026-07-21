from __future__ import annotations
from .layer import Layer
import random
import numpy as np
from numpy.typing import NDArray


class RMSNorm(Layer):
    def __init__(self: RMSNorm) -> None:
        super().__init__()
        self.gamma: float = 1
        self.rms: NDArray[np.float64] = np.array([[]])

    def feed_forward(self: RMSNorm, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        self.rms = np.sqrt(np.sum(entry**2, axis=0) / entry.shape[0] + 10e-12).reshape(1, -1)
        return self.gamma * entry / self.rms

    def descend_gradient(self: RMSNorm, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = gradient * self.gamma / self.rms * (1 - (self.input / self.rms) ** 2)
        self.gamma -= self.lr * sum((gradient * self.input / self.rms).flatten().tolist())
        return new_gradient
