from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .exit_loss import ExitLoss
from ..utils.functions import softmax


class ProbaExit(ExitLoss):
    def __init__(self: ProbaExit):
        super().__init__()

    def feed_forward(self: ProbaExit, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return softmax(entry)

    def get_loss(self: ProbaExit, prediction: NDArray[np.float64], answer: NDArray[np.float64]) -> float:
        epsilon = 1e-10
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.sum(answer * np.log(prediction))  # cross-entropy
        return loss

    def get_gradient(
        self: ProbaExit, prediction: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return prediction - answer  # cross-entropy + softmax
