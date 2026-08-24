from __future__ import annotations
import numpy as np
from ..utils.typing import Tensor, Receive1, SaveData, ShapeFlow
from ..basics.layer import Layer


class UpSample(Layer):
    def __init__(self: UpSample, factor: int = 2, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.factor = factor
    
    def set_input_shape(self: UpSample, input_shape: ShapeFlow) -> ShapeFlow:
        C, H, W = input_shape[0]
        super().set_input_shape(input_shape)
        self.output_shape = ((C, H * self.factor, W * self.factor),)
        return self.output_shape

    def feed_forward(self: UpSample, entry: Tensor) -> Tensor:
        return np.repeat(np.repeat(entry, self.factor, axis=1), self.factor, axis=2)

    def descend_gradient(self: UpSample, gradient: Tensor) -> Tensor:
        C, Hout, Wout = gradient.shape
        H = Hout // self.factor
        W = Wout // self.factor
        return gradient.reshape(C, H, self.factor, W, self.factor).sum(axis=(2, 4))

    def get_data(self: UpSample) -> dict:
        data = super().get_data()
        data["factor"] = self.factor
        return data

    def load_from_data(self: UpSample, data: SaveData) -> None:
        super().load_from_data(data)
        self.factor = data["factor"]
