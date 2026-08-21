from __future__ import annotations
from ..basics.layer import Layer
import numpy as np
from ..utils.typing import Tensor, Receive1


class Activation(Layer):
    def __init__(self: Activation, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

    def feed_forward(self: Activation, entry: Tensor) -> Tensor:
        return entry

    def compute_derivative(self: Activation, entry: Tensor) -> Tensor:
        return np.ones_like(entry)

    def descend_gradient(self: Activation, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        return gradient * self.compute_derivative(self.input[0])
