from __future__ import annotations
from ..utils.typing import Tensor, Receive1, SaveData, ShapeFlow
from ..basics.layer import Layer
import numpy as np


class SinEmbedding(Layer):
    def __init__(self: SinEmbedding, dim: int = 1, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.dim: int = dim

    def set_input_shape(self: SinEmbedding, input_shape: ShapeFlow) -> ShapeFlow:
        if input_shape != ((1, 1),):
            raise ValueError
        super().set_input_shape(input_shape)
        self.output_shape = ((self.dim, 1),)
        return self.output_shape

    def feed_forward(self: SinEmbedding, entry: Tensor) -> Tensor:
        if entry.shape != (1, 1):
            raise ValueError(f"Expected input shape (1, 1), got {entry.shape}")
        t = entry[0, 0]
        half_dim = self.dim // 2
        indices = np.arange(half_dim)
        frequencies = 10000 ** (-2 * indices / self.dim)
        angles = t * frequencies

        embedding = np.empty(self.dim)
        embedding[0 : 2 * half_dim : 2] = np.sin(angles)
        embedding[1 : 2 * half_dim : 2] = np.cos(angles)

        # Si dim est impair, on laisse le dernier terme à 0.
        if self.dim % 2 == 1:
            embedding[-1] = 0.0
        return embedding.reshape(self.dim, 1)

    def descend_gradient(self: SinEmbedding, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError

        t = self.input[0][0, 0]
        half_dim = self.dim // 2
        indices = np.arange(half_dim)
        frequencies = 10000 ** (-2 * indices / self.dim)
        angles = t * frequencies

        # d sin(t*f) / dt = f*cos(t*f)
        # d cos(t*f) / dt = -f*sin(t*f)
        d_embedding = np.empty(self.dim)
        d_embedding[0 : 2 * half_dim : 2] = frequencies * np.cos(angles)
        d_embedding[1 : 2 * half_dim : 2] = -frequencies * np.sin(angles)

        if self.dim % 2 == 1:
            d_embedding[-1] = 0.0
        # gradient : dL/dE
        # dL/dt = sum_i (dL/dE_i)(dE_i/dt)
        d_t = np.sum(gradient[:, 0] * d_embedding)
        return np.array([[d_t]])

    def get_data(self: SinEmbedding) -> SaveData:
        data = super().get_data()
        data["dim"] = self.dim
        return data

    def load_from_data(self: SinEmbedding, data: SaveData) -> None:
        super().load_from_data(data)
        self.dim = data["dim"]
