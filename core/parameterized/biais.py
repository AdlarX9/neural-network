from __future__ import annotations
import numpy as np
from ..basics.layer import Layer
from ..utils.typing import ShapeFlow, Tensor, SaveData, ParamGrad


class Biais(Layer):
    def __init__(self: Biais, receive: int = 0) -> None:
        super().__init__((receive,))
        self.B = np.array([[]])
        self.parameters = ["B"]

    def set_input_shape(self: Biais, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0][0]), size=(input_shape[0][0], 1))  # He
        return input_shape

    def feed_forward(self: Biais, entry: Tensor) -> Tensor:
        if len(entry.shape) == 2:
            return entry + self.B
        elif len(entry.shape) == 3:
            return entry + np.expand_dims(self.B, axis=2)
        else:
            raise ValueError

    def descend_gradient(self: Biais, gradient: Tensor) -> Tensor:
        return gradient

    def params_gradient(self: Biais, gradient) -> ParamGrad:
        if self.input is None:
            raise MemoryError
        if len(self.input[0].shape) == 2:
            return {"B": np.sum(gradient, axis=1, keepdims=True)}
        elif len(self.input[0].shape) == 3:
            B_gradient = np.zeros_like(self.B)
            for i in range(self.B.shape[0]):
                B_gradient[i, 0] += np.sum(gradient[i, :, :])
            return {"B": B_gradient}
        else:
            raise ValueError
