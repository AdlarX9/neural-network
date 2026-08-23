from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..basics.layer import Layer
from ..utils.typing import Receive1, Tensor, ShapeFlow, SaveData


class Dropout(Layer):
    def __init__(self: Dropout, p: float = 0.1, inverted: bool = False, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        if not 0 <= p < 1:
            raise ValueError("p must satisfy 0 <= p < 1")

        self.p = p
        self.inverted: bool = inverted
        self.mask: NDArray[np.bool] | None = None

    def set_input_shape(
        self: Dropout,
        input_shape: ShapeFlow,
    ) -> ShapeFlow:
        return super().set_input_shape(input_shape)

    def feed_forward(
        self: Dropout,
        entry: Tensor,
    ) -> Tensor:
        # Si compute(..., memorize=False) est utilisé pour l'inférence,
        # on ne fait pas de dropout.
        if self.input is None:
            self.mask = None
            return entry

        self.mask = np.random.random(entry.shape) >= self.p
        if self.mask is None:
            raise ValueError
        if self.inverted:
            return entry * self.mask / (1 - self.p)
        return entry * self.mask

    def descend_gradient(
        self: Dropout,
        gradient: Tensor,
    ) -> Tensor:
        if self.mask is None:
            raise MemoryError("Dropout mask is unavailable")
        if self.inverted:
            return gradient * self.mask / (1 - self.p)
        return gradient * self.mask

    def get_data(self: Dropout) -> SaveData:
        data = super().get_data()
        data["inverted"] = self.inverted
        data["p"] = self.p
        return data

    def load_from_data(self: Dropout, data: SaveData) -> None:
        super().load_from_data(data)
        self.inverted = data["inverted"]
        self.p = data["p"]
