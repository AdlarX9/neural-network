from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from ..utils.typing import ShapeFlow, Tensor, Receive1, SaveData


class RMSNorm(Layer):
    def __init__(self: RMSNorm, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.gamma: Tensor = np.array([[]])
        self.rms: Tensor = np.array([[]])

    def set_input_shape(self: RMSNorm, input_shape: ShapeFlow) -> ShapeFlow:
        self.gamma = np.ones((input_shape[0][0], 1))
        return super().set_input_shape(input_shape)

    def feed_forward(self: RMSNorm, entry: Tensor) -> Tensor:
        self.rms = np.sqrt(np.sum(entry**2, axis=0) / entry.shape[0] + 1e-12).reshape(1, -1)
        return self.gamma * entry / self.rms

    def descend_gradient(self: RMSNorm, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        d = self.input[0].shape[0]
        dot = np.sum(gradient * self.input[0], axis=0, keepdims=True)
        new_gradient = self.gamma / self.rms * (gradient - self.input[0] * dot / (d * self.rms**2))

        self.gamma -= self.lr * np.sum(gradient * self.input[0] / self.rms, axis=1, keepdims=True)
        return new_gradient

    def get_data(self: RMSNorm) -> SaveData:
        data = super().get_data()
        data["gamma"] = self.gamma.flatten().tolist()
        return data

    def load_from_data(self: RMSNorm, data: SaveData) -> None:
        super().load_from_data(data)
        self.gamma = np.array(data["gamma"]).reshape(self.input_shape[0][0], 1)
