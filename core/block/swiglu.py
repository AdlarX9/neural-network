from __future__ import annotations
from .block import Block
from ..transform.duplicate import Duplicate
from ..transform.multiply import Multiply
from .linear import Linear
from ..activation.silu import SiLU
from ..layer.layer import Layer
from ..utils.typing import Receive1


class SwiGLU(Block):
    def __init__(self: SwiGLU, n: int = 1, receive: Receive1 = (0,)) -> None:
        layers: list[Layer] = [
            Duplicate(2),
            Linear(n, receive=(0,)),
            SiLU(receive=(0,)),
            Linear(n, receive=(1,)),
            Multiply(receive=(0, 1)),
        ]
        super().__init__(layers, receive)
