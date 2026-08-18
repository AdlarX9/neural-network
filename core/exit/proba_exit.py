from __future__ import annotations
from core.layer.layer import Layer
import numpy as np
from numpy.typing import NDArray
from .exit_loss import ExitLoss
from ..utils.functions import softmax


class ProbaExit(ExitLoss):
    def __init__(self: ProbaExit, axis: int | None = None, receive: int = 0):
        self.axis = axis
        super().__init__(receive)

    def feed_forward(self: ProbaExit, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return softmax(entry, axis=self.axis)

    def get_loss(self: ProbaExit, prediction: NDArray[np.float64], answer: NDArray[np.float64]) -> float:
        epsilon = 1e-10
        _, p = prediction.shape
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.sum(answer * np.log(prediction)) / p  # cross-entropy
        return loss

    def get_gradient(
        self: ProbaExit, prediction: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return prediction - answer  # cross-entropy + softmax

    def get_data(self: ProbaExit) -> dict:
        data = super().get_data()
        data["axis"] = self.axis
        return data

    def load_from_data(self: ProbaExit, data: dict) -> None:
        super().load_from_data(data)
        self.axis = data["axis"]
