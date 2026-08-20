from __future__ import annotations
import numpy as np
from .activation import Activation
from ..utils.functions import sigmoid
from ..utils.typing import Tensor, Receive1


class SiLU(Activation):
    def __init__(self: SiLU, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

    def feed_forward(self: SiLU, entry: Tensor) -> Tensor:
        return entry * sigmoid(entry)

    def compute_derivative(self: SiLU, entry: Tensor) -> Tensor:
        return sigmoid(entry) * (np.ones_like(entry) + entry - self.feed_forward(entry))
