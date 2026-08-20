from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from ..utils.typing import ShapeFlow, Tensor, TensorFlow, Receive1, SaveData


class Duplicate(Layer):
    def __init__(self: Duplicate, factor: int = 1, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.factor: int = factor

    def set_input_shape(self: Duplicate, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        self.output_shape = tuple([input_shape[0] for _ in range(self.factor)])
        return self.output_shape

    def feed_forward(self: Duplicate, entry: Tensor) -> TensorFlow:
        return tuple([entry.copy() for _ in range(self.factor)])

    def descend_gradient(self: Duplicate, gradient: TensorFlow) -> Tensor:
        if self.input is None:
            raise MemoryError
        new_gradient: Tensor = np.zeros_like(self.input[0])
        for grad in gradient:
            new_gradient += grad
        return new_gradient

    def get_data(self: Duplicate) -> SaveData:
        data = super().get_data()
        data["factor"] = self.factor
        return data

    def load_from_data(self: Duplicate, data: SaveData) -> None:
        super().load_from_data(data)
        self.factor = data["factor"]
