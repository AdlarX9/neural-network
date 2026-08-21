from __future__ import annotations
from ..basics.layer import Layer
from ..basics.block import Block
from ..parameterized.lstm import LSTM
import numpy as np
from ..utils.typing import ShapeFlow, TensorFlow, Receive1, ParamGrad


class Recurrent(Block):
    def __init__(self: Recurrent, layers: list[Layer] = [], receive: Receive1 = (0,)):
        super().__init__(layers, receive)

    def reset_data(self: Recurrent) -> None:
        for layer in self.layers:
            if isinstance(layer, LSTM):
                layer.reset_data()

    def set_input_shape(self: Recurrent, input_shape: ShapeFlow) -> ShapeFlow:
        n, _ = input_shape[0]
        super().set_input_shape(((n, 1),))
        self.input_shape = input_shape
        return self.output_shape

    def __call__(self: Recurrent, entry: TensorFlow, memorize: bool) -> TensorFlow:
        _, p = entry[0].shape
        out = (np.array([[]]),)
        self.reset_data()
        for i in range(p):
            vec = entry[0][:, i].reshape(-1, 1)
            out = super().__call__((vec,), memorize)
        if memorize:
            self.input = entry
        return out

    def backprop(self: Recurrent, gradient: TensorFlow) -> tuple[TensorFlow, ParamGrad]:
        if self.input is None:
            raise MemoryError
        _, p = self.input[0].shape
        new_gradient: TensorFlow | None = None
        params: ParamGrad | None = None
        for _ in reversed(range(p)):
            gradient, param = super().backprop(gradient)
            if params is None:
                params = param
            else:
                params = {key: params[key] + param[key] for key in params}
            if new_gradient is None:
                new_gradient = gradient
            else:
                new_gradient = (np.hstack((gradient[0], new_gradient[0])),)
        if new_gradient is None or params is None:
            raise ValueError
        return new_gradient, params
