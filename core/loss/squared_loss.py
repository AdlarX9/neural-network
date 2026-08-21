from __future__ import annotations
import numpy as np
from .loss import Loss
from ..utils.typing import TensorFlow


class SquaredLoss(Loss):
    def __init__(self: Loss) -> None:
        return

    def get_loss(self: SquaredLoss, prediction: TensorFlow, answer: TensorFlow) -> tuple[float, ...]:
        if len(prediction) != len(answer):
            raise ValueError("prediction and answer must have the same number of tensors")

        losses: list[float] = []
        for pred, ans in zip(prediction, answer):
            if pred.shape != ans.shape:
                raise ValueError("prediction and answer tensors must have matching shapes")
            losses.append(float(np.sum(0.5 * (pred - ans) ** 2)))
        return tuple(losses)

    def get_gradient(self: SquaredLoss, prediction: TensorFlow, answer: TensorFlow) -> TensorFlow:
        if len(prediction) != len(answer):
            raise ValueError("prediction and answer must have the same number of tensors")

        gradients = []
        for pred, ans in zip(prediction, answer):
            if pred.shape != ans.shape:
                raise ValueError("prediction and answer tensors must have matching shapes")
            gradients.append(pred - ans)
        return tuple(gradients)
