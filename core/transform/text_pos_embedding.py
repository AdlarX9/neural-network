from __future__ import annotations
import numpy as np
from ..basics.layer import Layer
from ..utils.typing import Receive1, Tensor, ShapeFlow


class TextPosEmbedding(Layer):
    def __init__(self: TextPosEmbedding, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)

        self.position_embedding: Tensor = np.array([[]])

    def set_input_shape(self: TextPosEmbedding, input_shape: ShapeFlow) -> ShapeFlow:
        d, T = input_shape[0]
        if d % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {d}")

        positions = np.arange(T)[:, None]
        dimensions = np.arange(0, d, 2)[None, :]
        frequencies = 10000 ** (-dimensions / d)
        angles = positions * frequencies

        self.position_embedding = np.zeros((d, T))
        self.position_embedding[0::2, :] = np.sin(angles).T
        self.position_embedding[1::2, :] = np.cos(angles).T

        super().set_input_shape(input_shape)
        return self.output_shape

    def feed_forward(self: TextPosEmbedding, entry: Tensor) -> Tensor:
        return entry + self.position_embedding

    def descend_gradient(self: TextPosEmbedding, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError

        return gradient
