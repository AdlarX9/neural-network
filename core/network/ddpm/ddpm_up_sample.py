from __future__ import annotations
from ...basics.layer import Layer
from ...basics.block import Block
from ...parameterized.conv import Conv
from ...transform.up_sample import UpSample
from ...flowmakers.concat import Concat
from ...utils.typing import Receive2


class DDPMUpSample(Block):
    def __init__(self: DDPMUpSample, previous_C: int = 0, receive: Receive2 = (0, 1)) -> None:
        """
        0: skip connection -> 0: image agrandie concaténée (2 * previous_C, 2H, 2W)
        1: image à agrandir (C, H, W)
        """
        layers: list[Layer] = [
            UpSample(factor=2, receive=(1,)),
            Conv(N=previous_C, K=3, receive=(1,)),
            Concat(axis=0, receive=(0, 1)),
        ]
        super().__init__(layers, receive)
