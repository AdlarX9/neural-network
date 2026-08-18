from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class ExitLoss(Layer):
    def __init__(self: ExitLoss, receive: int = 0):
        super().__init__((receive,))

    def get_loss(self: ExitLoss, prediction: NDArray[np.float64], answer: NDArray[np.float64]) -> float:
        return np.sum(1 / 2 * (prediction - answer) ** 2)  # quadratic loss

    def get_gradient(
        self: ExitLoss, prediction: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return prediction - answer
