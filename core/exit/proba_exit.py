from __future__ import annotations
from core.layer.layer import Layer
import numpy as np
from numpy.typing import NDArray
from .exit_loss import ExitLoss
from ..utils.functions import softmax


class ProbaExit(ExitLoss):
    def __init__(self: ProbaExit, axis: int | None = None):
        self.axis = axis
        super().__init__()

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

    def get_data(self: ProbaExit) -> tuple[list[int], list[float], list[str]]:
        int_list, float_list, str_list = super().get_data()
        axis = self.axis
        if axis is None:
            axis = -1
        int_list.append(axis)
        return int_list, float_list, str_list

    def load_from_data(
        self: ProbaExit, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.axis = int_list.pop()
        if self.axis == -1:
            self.axis = None
        return super().load_from_data(int_list, float_list, string_list)
