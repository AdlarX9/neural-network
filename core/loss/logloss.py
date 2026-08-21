from __future__ import annotations
import numpy as np
from .loss import Loss
from ..utils.typing import TensorFlow


class LogLoss(Loss):
    def __init__(self: Loss) -> None:
        return

    def get_loss(self: LogLoss, prediction: TensorFlow, answer: TensorFlow) -> tuple[float, ...]:
        if len(prediction) != len(answer):
            raise ValueError("prediction and answer must have the same number of tensors")

        epsilon = 1e-10
        losses: list[float] = []
        for pred, ans in zip(prediction, answer):
            if pred.shape != ans.shape:
                raise ValueError("prediction and answer tensors must have matching shapes")
            p = pred.shape[-1] if pred.ndim > 0 else 1
            pred_clipped = np.clip(pred, epsilon, 1 - epsilon)
            loss = -np.sum(ans * np.log(pred_clipped)) / p
            losses.append(float(loss))
        return tuple(losses)

    def get_gradient(self: LogLoss, prediction: TensorFlow, answer: TensorFlow) -> TensorFlow:
        if len(prediction) != len(answer):
            raise ValueError("prediction and answer must have the same number of tensors")

        epsilon = 1e-10
        gradients = []
        for pred, ans in zip(prediction, answer):
            if pred.shape != ans.shape:
                raise ValueError("prediction and answer tensors must have matching shapes")
            p = pred.shape[-1] if pred.ndim > 0 else 1
            pred_clipped = np.clip(pred, epsilon, 1 - epsilon)
            gradients.append(-(ans / pred_clipped) / p)
        return tuple(gradients)
