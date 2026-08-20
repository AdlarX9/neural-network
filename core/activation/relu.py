from __future__ import annotations
import numpy as np
from ..utils.typing import Tensor, Receive1
from .activation import Activation


class ReLU(Activation):
    def __init__(self: ReLU, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

    def feed_forward(self: ReLU, entry: Tensor) -> Tensor:
        return np.maximum(0, entry)

    def compute_derivative(self: ReLU, entry: Tensor) -> Tensor:
        deriv = np.ones_like(entry)
        deriv[entry <= 0] = 0
        return deriv
