from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class AverageFlatten(Layer):
    def __init__(self: AverageFlatten) -> None:
        super().__init__()

    def set_input_shape(self: AverageFlatten, input_shape: tuple[int, int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        return (input_shape[0], 1)

    def feed_forward(self: AverageFlatten, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        out = []
        for i in range(entry.shape[0]):
            _, n, p = entry.shape
            out.append(np.sum(entry[i, :, :]) / (n * p))
        return np.array(out).reshape(-1, 1)

    def descend_gradient(self: AverageFlatten, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = np.zeros_like(self.input)
        for i in range(gradient.shape[0]):
            _, n, p = new_gradient.shape
            new_gradient[i, :, :] = np.full((n, p), gradient[i, 0] / (n * p))
        return new_gradient


class Flatten(Layer):
    def __init__(self: Flatten) -> None:
        return

    def set_input_shape(self: Flatten, input_shape: tuple[int, int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        c, n, p = input_shape
        return (c * n * p, 1)

    def feed_forward(self: Flatten, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry.reshape(-1, 1)

    def descend_gradient(self: Flatten, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        return gradient.reshape(self.input.shape)
