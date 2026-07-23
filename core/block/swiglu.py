from __future__ import annotations
from .block import Block
from .multiplier import Multiplier
from .linear import Linear
from ..activation.silu import SiLU


class SwiGLU(Block):
    def __init__(self: SwiGLU, n: int = 1) -> None:
        self.layers = [Multiplier(Block(Linear(n), SiLU()), Linear(n))]
        super().__init__()
