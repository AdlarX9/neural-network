from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Causal(Layer):
    def __init__(self: Causal, receive: tuple[int] = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(self: Causal, input_shape: tuple) -> tuple:
        super().set_input_shape(input_shape)
        if (
            len(input_shape[0]) == 2
            and input_shape[0][0] != input_shape[0][1]
            or len(input_shape[0]) == 3
            and input_shape[0][1] != input_shape[0][2]
        ):
            raise ValueError("Causal only accepts square matrices:", input_shape)
        return self.output_shape

    def feed_forward(self: Causal, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        mask = np.tril(np.full(entry.shape, -np.inf), k=-1)
        return entry + mask
