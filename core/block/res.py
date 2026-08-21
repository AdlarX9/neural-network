from __future__ import annotations
from ..basics.layer import Layer
from ..basics.block import Block
from ..flowmakers.duplicate import Duplicate
from ..flowmakers.add import Add
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
