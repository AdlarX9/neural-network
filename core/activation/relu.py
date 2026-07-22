from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .activation import Activation


class ReLU(Activation):
    def __init__(self: ReLU) -> None:
        super().__init__()

    def feed_forward(self: ReLU, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, entry)

    def compute_derivative(self: ReLU, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        deriv = np.ones_like(entry)
        deriv[entry <= 0] = 0
        return deriv
