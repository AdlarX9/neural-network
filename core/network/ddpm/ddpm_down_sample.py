from __future__ import annotations
from ...basics.layer import Layer
from ...basics.block import Block
from ...parameterized.conv import Conv
from ...flowmakers.duplicate import Duplicate
from ...utils.typing import Receive1, ShapeFlow


class DDPMDownSample(Block):
    def __init__(self: DDPMDownSample, receive: Receive1 = (0,)) -> None:
        """
        0: image à rétrécir -> | 0: image originale
                               | 1: image rétréci
        """
        super().__init__([], receive)

    def _get_layers(self: DDPMDownSample, shape: ShapeFlow) -> list[Layer]:
        C, H, W = shape[0]
        layers: list[Layer] = [
            Duplicate(factor=2, receive=(0,)),
            Conv(N=2 * C, K=3, S=2, P=1, receive=(1,)),
        ]
        return layers

    def set_input_shape(self: DDPMDownSample, input_shape: ShapeFlow) -> ShapeFlow:
        self.layers = self._get_layers(input_shape)
        return super().set_input_shape(input_shape)
