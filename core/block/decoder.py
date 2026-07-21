from __future__ import annotations
import numpy as np
from .block import Block
from ..layer.fc import FC
from ..layer.biais import Biais
from ..layer.embedding import Embedding
from ..layer.layer import check_shapes
from numpy.typing import NDArray


class Decoder(Block):
    def __init__(self: Decoder, embedding: Embedding | None = None) -> None:
        if embedding is None:
            super().__init__()
            return
        super().__init__(FC(embedding.input_shape[0]), Biais())
        self.input_shape = embedding.output_shape
        self.output_shape = embedding.input_shape
        fc = self.layers[0]
        if isinstance(fc, FC):
            fc.W = embedding.W_prime.copy()
            fc.input_shape = self.input_shape
            fc.output_shape = self.output_shape
            fc.n = fc.output_shape[0]
            fc.p = fc.input_shape[0]
        else:
            raise ValueError
        biais = self.layers[1]
        if isinstance(biais, Biais):
            biais.B = np.random.normal(
                0, np.sqrt(2 / self.output_shape[0]), size=(self.output_shape[0], 1)
            )  # He
            biais.input_shape = self.output_shape
            biais.output_shape = self.output_shape
        else:
            raise ValueError

    def set_input_shape(self: Decoder, input_shape: tuple) -> tuple:
        if not check_shapes(input_shape, self.input_shape):
            raise ValueError("mismatch in input shapes:", input_shape, self.input_shape)
        return self.output_shape
    
    # def backprop(self: Decoder, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    #     return self.layers[0].W.T @ gradient
