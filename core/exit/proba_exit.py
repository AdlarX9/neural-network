from __future__ import annotations
import numpy as np
from .exit_loss import ExitLoss
from ..utils.functions import softmax
from ..utils.typing import Tensor, Receive1, SaveData


class ProbaExit(ExitLoss):
    def __init__(self: ProbaExit, axis: int | None = None, receive: Receive1 = (0,)):
        self.axis = axis
        super().__init__(receive)

    def feed_forward(self: ProbaExit, entry: Tensor) -> Tensor:
        return softmax(entry, axis=self.axis)

    def get_loss(self: ProbaExit, prediction: Tensor, answer: Tensor) -> float:
        epsilon = 1e-10
        _, p = prediction.shape
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.sum(answer * np.log(prediction)) / p  # cross-entropy
        return loss

    def get_gradient(self: ProbaExit, prediction: Tensor, answer: Tensor) -> Tensor:
        _, p = prediction.shape
        return (prediction - answer) / p  # cross-entropy + softmax

    def get_data(self: ProbaExit) -> SaveData:
        data = super().get_data()
        data["axis"] = self.axis
        return data

    def load_from_data(self: ProbaExit, data: SaveData) -> None:
        super().load_from_data(data)
        self.axis = data["axis"]
