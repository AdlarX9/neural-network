from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer
from ..utils.typing import Tensor, Receive1


class ExitLoss(Layer):
    def __init__(self: ExitLoss, receive: Receive1 = (0,)):
        super().__init__(receive)

    def get_loss(self: ExitLoss, prediction: Tensor, answer: Tensor) -> float:
        return np.sum(1 / 2 * (prediction - answer) ** 2)  # quadratic loss

    def get_gradient(self: ExitLoss, prediction: Tensor, answer: Tensor) -> Tensor:
        return prediction - answer
