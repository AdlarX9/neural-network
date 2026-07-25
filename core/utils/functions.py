import numpy as np
from numpy.typing import NDArray


def softmax(entry: NDArray[np.float64], axis: int | None = None) -> NDArray[np.float64]:
    max_entry = np.max(entry, axis=axis, keepdims=True)
    exp_entry = np.exp(entry - max_entry)
    softmax = exp_entry / np.sum(exp_entry, axis=axis, keepdims=True)
    return softmax


def sigmoid(entry: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-entry))


def RoPE(x: NDArray[np.float64], factor: int = 1) -> NDArray[np.float64]:
    _, n, p = x.shape
    if n % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {n}")
    rotated = np.empty_like(x)
    rotated[:, 0::2, :] = -x[:, 1::2, :]
    rotated[:, 1::2, :] = x[:, 0::2, :]
    positions = np.arange(p)[None, :]
    omega = 10000 ** (-2 * (np.arange(n) // 2) / n)[:, None]
    theta = omega @ positions
    theta = theta[None, :, :]
    return x * np.cos(factor * theta) + rotated * np.sin(factor * theta)
