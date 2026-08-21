from __future__ import annotations
from ..utils.typing import Tensor, Receive1, SaveData
from ..basics.layer import Layer


class Scale(Layer):
    def __init__(self: Scale, factor: float = 1, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.factor: float = factor

    def feed_forward(self: Scale, entry: Tensor) -> Tensor:
        return self.factor * entry

    def descend_gradient(self: Scale, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        return self.factor * gradient

    def get_data(self: Scale) -> SaveData:
        data = super().get_data()
        data["factor"] = self.factor
        return data

    def load_from_data(self: Scale, data: SaveData) -> None:
        super().load_from_data(data)
        self.factor = data["factor"]
