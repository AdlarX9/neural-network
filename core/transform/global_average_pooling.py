from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer


class GlobalAveragePooling(Layer):
    def __init__(self: GlobalAveragePooling, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(
        self: GlobalAveragePooling,
        input_shape: tuple[tuple[int, int, int]],
    ) -> tuple[tuple[int, int]]:
        super().set_input_shape(input_shape)
        C, H, W = input_shape[0]
        self.output_shape = ((C, 1),)
        return self.output_shape

    def feed_forward(self: GlobalAveragePooling, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.mean(entry, axis=(1, 2), keepdims=False).reshape(-1, 1)

    def descend_gradient(self: GlobalAveragePooling, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        C, H, W = self.input[0].shape
        return np.broadcast_to(gradient / (H * W), (C, H, W)).copy()
