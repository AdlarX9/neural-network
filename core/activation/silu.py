from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .activation import Activation
from ..utils.functions import sigmoid


class SiLU(Activation):
    def __init__(self: SiLU) -> None:
        super().__init__()

    def feed_forward(self: SiLU, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry * sigmoid(entry)

    def compute_derivative(self: SiLU, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return sigmoid(entry) * (np.ones_like(entry) + entry - self.feed_forward(entry))
