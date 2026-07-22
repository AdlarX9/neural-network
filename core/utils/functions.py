import numpy as np
from numpy.typing import NDArray


def softmax(entry: NDArray[np.float64], axis: int | None = None) -> NDArray[np.float64]:
    max_entry = np.max(entry, axis=axis, keepdims=True)
    exp_entry = np.exp(entry - max_entry)
    softmax = exp_entry / np.sum(exp_entry, axis=axis, keepdims=True)
    return softmax


def sigmoid(entry: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-entry))
