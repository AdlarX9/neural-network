from __future__ import annotations
from typing import Any
from ..basics.layer import Layer
import numpy as np
from ..utils.typing import ShapeFlow, Tensor, Receive1, SaveData, ParamGrad


class BN(Layer):
    def __init__(self: BN, receive: Receive1 = (0,)):
        super().__init__(receive)
        self.gamma: Tensor = np.array([[]])
        self.beta: Tensor = np.array([[]])
        self.epsilon = 1e-5
        self.mean = np.var([[]])
        self.var = np.var([[]])
        self.x_hat = np.var([[]])
        self.parameters = ["gamma", "beta"]

    def set_input_shape(self: BN, input_shape: ShapeFlow) -> ShapeFlow:
        C, _, _ = input_shape[0]
        self.gamma = np.ones((C, 1))
        self.beta = np.zeros((C, 1))
        super().set_input_shape(input_shape)
        return self.output_shape

    def feed_forward(self: BN, entry: Tensor) -> Tensor:
        # moyenne par canal
        self.mean = np.mean(entry, axis=(1, 2), keepdims=True)
        # variance par canal
        self.var = np.var(entry, axis=(1, 2), keepdims=True)
        self.x_hat = (entry - self.mean) / np.sqrt(self.var + self.epsilon)
        output = self.gamma[:, None, None] * self.x_hat + self.beta[:, None, None]
        return output

    def descend_gradient(self: BN, gradient: Tensor) -> Tensor:
        if self.input is None or self.x_hat is None or self.gamma is None or self.beta is None:
            raise MemoryError
        _, H, W = gradient.shape
        m = H * W

        dx_hat = gradient * self.gamma[:, None, None]
        sum_dxhat = np.sum(dx_hat, axis=(1, 2), keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=(1, 2), keepdims=True)
        dx = 1 / np.sqrt(self.var + self.epsilon) * (dx_hat - sum_dxhat / m - self.x_hat * sum_dxhat_xhat / m)
        return dx

    def params_gradient(self: BN, gradient) -> ParamGrad:
        return {
            "gamma": np.sum(gradient * self.x_hat, axis=(1, 2)),
            "beta": np.sum(gradient, axis=(1, 2)),
        }
