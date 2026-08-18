from __future__ import annotations
from ..layer.layer import Layer
from .block import Block
from ..transform.duplicate import Duplicate
from ..transform.add import Add


class Res(Block):
    def __init__(self: Res, layer: Layer = Layer(), receive: tuple[int] = (0,)) -> None:
        self.layer = layer
        layers = [
            Duplicate(2),
            layer,
            Add(receive=(0, 1)),
        ]
        super().__init__(layers, receive)
