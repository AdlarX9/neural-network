from __future__ import annotations
import numpy as np
from ..basics.block import Block
from ..parameterized.fc import FC
from ..parameterized.biais import Biais
from ..parameterized.embedding import Embedding
from ..basics.layer import check_shapes
from ..utils.typing import ShapeFlow, Receive1


class OneHotMaker(Block):
    def __init__(self: OneHotMaker, embedding: Embedding | None = None, receive: Receive1 = (0,)) -> None:
        if embedding is None:
            super().__init__(receive=receive)
            return
        super().__init__([FC(embedding.input_shape[0][0]), Biais()], receive)
        self.input_shape = embedding.output_shape
        self.output_shape = embedding.input_shape
        fc = self.layers[0]
        if isinstance(fc, FC):
            fc.W = embedding.W_prime.copy()
            fc.input_shape = self.input_shape
            fc.output_shape = self.output_shape
            fc.n = fc.output_shape[0][0]
            fc.p = fc.input_shape[0][0]
        else:
            raise ValueError
        biais = self.layers[1]
        if isinstance(biais, Biais):
            biais.B = np.random.normal(
                0, np.sqrt(2 / self.output_shape[0][0]), size=(self.output_shape[0][0], 1)
            )  # He
            biais.input_shape = self.output_shape
            biais.output_shape = self.output_shape
        else:
            raise ValueError

    def set_input_shape(self: OneHotMaker, input_shape: ShapeFlow) -> ShapeFlow:
        if not check_shapes(input_shape, self.input_shape):
            raise ValueError("mismatch in input shapes:", input_shape, self.input_shape)
        return self.output_shape
