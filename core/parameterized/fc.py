from __future__ import annotations
import numpy as np
from ..basics.layer import Layer
from ..utils.typing import ShapeFlow, Tensor, Receive1, SaveData, ParamGrad


class FC(Layer):
    def __init__(self: FC, n: int = 0, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.n: int = n
        self.p: int = 0
        self.W = np.array([[]])
        self.parameters = ["W"]

    def set_input_shape(self: FC, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        p, q = input_shape[0]
        self.p = p
        self.W = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))  # He
        self.output_shape = ((self.n, q),)
        return self.output_shape

    def feed_forward(self: FC, entry: Tensor) -> Tensor:
        return self.W @ entry

    def descend_gradient(self: FC, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        return self.W.T @ gradient

    def params_gradient(self: FC, gradient) -> ParamGrad:
        if self.input is None:
            raise MemoryError
        return {"W": gradient @ self.input[0].T}

    def get_data(self: FC) -> SaveData:
        data = super().get_data()
        data["n"] = self.n
        data["p"] = self.p
        return data

    def load_from_data(self: FC, data: SaveData) -> None:
        super().load_from_data(data)
        self.n = data["n"]
        self.p = data["p"]
