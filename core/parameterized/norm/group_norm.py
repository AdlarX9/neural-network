from __future__ import annotations
import numpy as np
from ...basics.layer import Layer
from ...utils.typing import Receive1, Tensor, ShapeFlow, ParamGrad, SaveData


class GroupNorm(Layer):
    def __init__(self: GroupNorm, groups: int, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.groups = groups

        self.gamma: Tensor = np.array([[]])
        self.beta: Tensor = np.array([[]])

        self.epsilon = 1e-5
        self.mean: Tensor = np.array([[]])
        self.var: Tensor = np.array([[]])
        self.x_hat: Tensor = np.array([[]])

        self.parameters = ["gamma", "beta"]

    def set_input_shape(self: GroupNorm, input_shape: ShapeFlow) -> ShapeFlow:
        C = input_shape[0][0]
        if C % self.groups != 0:
            raise ValueError(
                f"Number of channels ({C}) must be divisible " f"by number of groups ({self.groups})"
            )

        # Gamma et beta : un paramètre par canal.
        shape = (C,) + (1,) * (len(input_shape[0]) - 1)
        self.gamma = np.ones(shape)
        self.beta = np.zeros(shape)
        return super().set_input_shape(input_shape)

    def feed_forward(
        self: GroupNorm,
        entry: Tensor,
    ) -> Tensor:
        C = entry.shape[0]
        channels_per_group = C // self.groups

        # (G, C/G, ...)
        grouped_shape = (self.groups, channels_per_group, *entry.shape[1:])
        x = entry.reshape(grouped_shape)

        # Toutes les dimensions du groupe sauf G.
        axes = tuple(range(1, x.ndim))
        self.mean = np.mean(x, axis=axes, keepdims=True)
        self.var = np.var(x, axis=axes, keepdims=True)
        x_hat = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        self.x_hat = x_hat.reshape(entry.shape)
        return self.gamma * self.x_hat + self.beta

    def descend_gradient(
        self: GroupNorm,
        gradient: Tensor,
    ) -> Tensor:
        if self.input is None:
            raise MemoryError
        entry = self.input[0]
        C = entry.shape[0]
        channels_per_group = C // self.groups
        grouped_shape = (self.groups, channels_per_group, *entry.shape[1:])
        dx_hat = (gradient * self.gamma).reshape(grouped_shape)
        x_hat = self.x_hat.reshape(grouped_shape)

        # Toutes les dimensions sauf la dimension des groupes.
        axes = tuple(range(1, dx_hat.ndim))
        m = np.prod([dx_hat.shape[i] for i in axes])
        sum_dxhat = np.sum(dx_hat, axis=axes, keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * x_hat, axis=axes, keepdims=True)
        dx = 1 / np.sqrt(self.var + self.epsilon) * (dx_hat - sum_dxhat / m - x_hat * sum_dxhat_xhat / m)
        return dx.reshape(entry.shape)

    def params_gradient(self: GroupNorm, gradient: Tensor) -> ParamGrad:
        return {
            "gamma": np.sum(
                gradient * self.x_hat,
                axis=tuple(range(1, gradient.ndim)),
                keepdims=True,
            ),
            "beta": np.sum(gradient, axis=tuple(range(1, gradient.ndim)), keepdims=True),
        }

    def get_data(self: GroupNorm) -> SaveData:
        data = super().get_data()
        data["groups"] = self.groups
        return data

    def load_from_data(self: GroupNorm, data: SaveData) -> None:
        super().load_from_data(data)
        self.groups = data["groups"]
