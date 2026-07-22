from __future__ import annotations
from .block import Block
from .multiplier import Multiplier
from ..layer.fc import FC
from ..activation.silu import SiLU


class SwiGLU(Block):
    def __init__(self: SwiGLU) -> None:
        super().__init__()

    def set_input_shape(self: SwiGLU, input_shape: tuple) -> tuple:
        n, _ = input_shape
        self.layers = [Multiplier(Block(FC(n), SiLU(), FC(n)), FC(n))]
        self.set_lr(self.lr)
        return super().set_input_shape(input_shape)
