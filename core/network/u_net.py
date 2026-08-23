from __future__ import annotations

from ..utils.typing import Receive, Shape, ShapeFlow
from ..basics.network import Network
from ..basics.block import Block
from ..basics.layer import Layer


class UNet(Network):
    def __init__(self: UNet, layers: list[Layer] = [], input_shape: ShapeFlow = ((0, ),), lr: float = 0.001, receive: Receive = (0, )) -> None:
        super().__init__(layers, input_shape, lr, receive)
