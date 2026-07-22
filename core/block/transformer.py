from __future__ import annotations
from .swiglu import SwiGLU
from ..layer.mha import MHA
from ..transform.rms_norm import RMSNorm
from .res import Res
from .block import Block


class TransformerBlock(Block):

    def __init__(self: TransformerBlock, H: int = 1):
        super().__init__()
        self.H = H

    def set_input_shape(self: TransformerBlock, input_shape: tuple[int, int]) -> tuple[int, int]:
        self.layers = [Res(Block(RMSNorm(), MHA(self.H))), Res(Block(RMSNorm(), SwiGLU()))]
        self.set_lr(self.lr)
        self._init = True
        return super().set_input_shape(input_shape)
