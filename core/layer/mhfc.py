from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from ..utils.typing import ShapeFlow, Tensor, Receive1, SaveData


class MHFC(Layer):
    """Stands for Multi-Head Fully Connected Layer"""

    def __init__(self: MHFC, H: int = 0, n: int = 0, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.H: int = H
        self.n: int = n
        self.p: int = 0
        self.W = np.array([[[]]])

    def set_input_shape(self: MHFC, input_shape:ShapeFlow) ->ShapeFlow:
        super().set_input_shape(input_shape)
        p, q = input_shape[0]
        self.p = p
        self.W = np.random.normal(0, np.sqrt(2 / self.n), size=(self.H, self.n, p))  # He
        self.output_shape = ((self.H, self.n, q),)
        return self.output_shape

    def feed_forward(self: MHFC, entry: Tensor) -> Tensor:
        return self.W @ entry

    def descend_gradient(self: MHFC, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        new_gradient = np.sum(self.W.swapaxes(1, 2) @ gradient, axis=0)
        self.W -= self.lr * gradient @ self.input[0].T
        return new_gradient

    def get_data(self: MHFC) -> SaveData:
        data = super().get_data()
        data["H"] = self.H
        data["n"] = self.n
        data["p"] = self.p
        data["W"] = self.W.flatten().tolist()
        return data

    def load_from_data(self: MHFC, data: SaveData) -> None:
        super().load_from_data(data)
        self.H = data["H"]
        self.n = data["n"]
        self.p = data["p"]
        self.W = np.array(data["W"]).reshape((self.H, self.n, self.p))
