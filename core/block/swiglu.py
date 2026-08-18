from __future__ import annotations
from .block import Block
from .multiplier import Multiplier
from .linear import Linear
from ..activation.silu import SiLU
from ..layer.layer import Layer


class SwiGLU(Block):
    def __init__(self: SwiGLU, n: int = 1, receive: tuple[int] = (0,)) -> None:
        layers: list[Layer] = [Multiplier([Block([Linear(n), SiLU()]), Linear(n)])]  # type: ignore
        super().__init__(layers, receive)
