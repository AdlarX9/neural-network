from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class MaxPooling(Layer):
    def __init__(self: MaxPooling, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self._max_pos: NDArray[np.int64] | None = None  # positions des max dans chaque fenêtre

    def feed_forward(self: MaxPooling, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        c, n, p = entry.shape
        d = self.dim

        out_h = math.ceil(n / d)
        out_w = math.ceil(p / d)

        # Padding à -inf pour que le max ne soit pas biaisé sur les bords
        pad_h = out_h * d - n
        pad_w = out_w * d - p
        x = np.pad(
            entry,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=-np.inf,
        )  # (c, out_h*d, out_w*d)

        # Regroupement en blocs (c, out_h, d, out_w, d)
        x_blocks = x.reshape(c, out_h, d, out_w, d).transpose(0, 1, 3, 2, 4)
        flat = x_blocks.reshape(c, out_h, out_w, d * d)

        # Max pooling vectorisé
        self._max_pos = np.argmax(flat, axis=-1)  # (c, out_h, out_w)
        out = np.max(flat, axis=-1)  # (c, out_h, out_w)
        return out

    def descend_gradient(self: MaxPooling, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None or self._max_pos is None:
            raise MemoryError("feed_forward must be called before descend_gradient")

        c, n, p = self.input.shape
        d = self.dim
        out_h, out_w = gradient.shape[1], gradient.shape[2]

        pad_h = out_h * d - n
        pad_w = out_w * d - p

        # Gradient dans les blocs paddés
        grad_blocks = np.zeros((c, out_h, out_w, d * d), dtype=self.input.dtype)

        # Indices vectorisés pour placer chaque gradient sur la position du max
        ci = np.arange(c)[:, None, None]
        hi = np.arange(out_h)[None, :, None]
        wi = np.arange(out_w)[None, None, :]
        grad_blocks[ci, hi, wi, self._max_pos] = gradient

        # Retour en image (avec crop pour enlever padding)
        grad_blocks = grad_blocks.reshape(c, out_h, out_w, d, d).transpose(0, 1, 3, 2, 4)
        grad_padded = grad_blocks.reshape(c, out_h * d, out_w * d)

        if pad_h > 0:
            grad_padded = grad_padded[:, :-pad_h, :]
        if pad_w > 0:
            grad_padded = grad_padded[:, :, :-pad_w]

        return grad_padded

    def get_data(self: MaxPooling) -> dict:
        data = super().get_data()
        data["dim"] = self.dim
        return data

    def load_from_data(self: MaxPooling, data: dict) -> None:
        super().load_from_data(data)
        self.dim = data["dim"]
