import numpy as np
from .typing import Tensor


def softmax(entry: Tensor, axis: int | None = None) -> Tensor:
    max_entry = np.max(entry, axis=axis, keepdims=True)
    exp_entry = np.exp(entry - max_entry)
    softmax = exp_entry / np.sum(exp_entry, axis=axis, keepdims=True)
    return softmax


def sigmoid(entry: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-entry))
