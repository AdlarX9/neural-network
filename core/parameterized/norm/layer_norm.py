from __future__ import annotations
import numpy as np
from ...basics.layer import Layer
from ...utils.typing import Receive1, Tensor, ShapeFlow, ParamGrad


class LayerNorm(Layer):
    def __init__(self: LayerNorm, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

        self.gamma: Tensor = np.array([[]])
        self.beta: Tensor = np.array([[]])

        self.epsilon = 1e-5
        self.mean: Tensor = np.array([[]])
        self.var: Tensor = np.array([[]])
        self.x_hat: Tensor = np.array([[]])

        self.parameters = ["gamma", "beta"]

    def set_input_shape(self: LayerNorm, input_shape: ShapeFlow) -> ShapeFlow:
        shape = input_shape[0]
        # Un paramètre par élément du tenseur.
        self.gamma = np.ones(shape)
        self.beta = np.zeros(shape)
        return super().set_input_shape(input_shape)

    def feed_forward(self: LayerNorm, entry: Tensor) -> Tensor:
        # On normalise chaque tenseur indépendamment sur toutes ses dimensions.
        axes = tuple(range(entry.ndim))
        self.mean = np.mean(entry, axis=axes, keepdims=True)
        self.var = np.var(entry, axis=axes, keepdims=True)
        self.x_hat = (entry - self.mean) / np.sqrt(self.var + self.epsilon)
        return self.gamma * self.x_hat + self.beta

    def descend_gradient(
        self: LayerNorm,
        gradient: Tensor,
    ) -> Tensor:
        if self.input is None:
            raise MemoryError
        entry = self.input[0]
        axes = tuple(range(entry.ndim))
        m = entry.size
        dx_hat = gradient * self.gamma
        sum_dxhat = np.sum(dx_hat, axis=axes, keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=axes, keepdims=True)
        dx = 1 / np.sqrt(self.var + self.epsilon) * (dx_hat - sum_dxhat / m - self.x_hat * sum_dxhat_xhat / m)
        return dx

    def params_gradient(self: LayerNorm, gradient: Tensor) -> ParamGrad:
        return {
            "gamma": gradient * self.x_hat,
            "beta": gradient,
        }
