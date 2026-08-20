from __future__ import annotations
from ..layer.layer import Layer
from .block import Block
from ..transform.duplicate import Duplicate
from ..transform.add import Add
from ..utils.typing import Receive1


class Res(Block):
    def __init__(self: Res, layer: Layer = Layer(), receive: Receive1 = (0,)) -> None:
        self.layer = layer
        layers = [
            Duplicate(2),
            layer,
            Add(receive=(0, 1)),
        ]
        super().__init__(layers, receive)
