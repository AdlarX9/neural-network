from __future__ import annotations
from ...basics.layer import Layer
from ...basics.block import Block
from ...flowmakers.duplicate import Duplicate
from ...flowmakers.add import Add
from ...utils.typing import Receive2, ShapeFlow, SaveData
from ...transform.dropout import Dropout
from ...parameterized.conv import Conv
from ...basics.layer import Layer
from ...activation.silu import SiLU
from ...basics.block import Block
from ...block.linear import Linear
from ...flowmakers.duplicate import Duplicate
from ...flowmakers.add import Add
from ...transform.reshape import Reshape
from ...parameterized.norm.group_norm import GroupNorm


class DDPMResBlock(Block):
    def __init__(self: DDPMResBlock, C: int = -1, receive: Receive2 = (0, 1)) -> None:
        """
        0: image          | -> 0: image transformée
        1: time embedding |
        """
        self.C = C
        super().__init__([], receive)

    def _get_layers(self: DDPMResBlock, shape: ShapeFlow) -> list[Layer]:
        C = shape[0][0]
        if self.C == -1:
            self.C = C
        layers: list[Layer] = [
            Duplicate(factor=2, receive=(0,)),
            GroupNorm(groups=32, receive=(0,)),
            SiLU(receive=(0,)),
            Conv(N=self.C, K=3, receive=(0,)),
            Linear(neuron_number=self.C, receive=(2,)),
            Reshape(shape=(self.C, 1, 1), receive=(2,)),
            Add(receive=(0, 2)),
            GroupNorm(groups=32, receive=(0,)),
            SiLU(receive=(0,)),
            Dropout(p=0.1, inverted=True, receive=(0,)),
            Conv(N=self.C, K=3, receive=(0,)),
            Add(receive=(0, 1)),
        ]
        return layers

    def set_input_shape(self: DDPMResBlock, input_shape: ShapeFlow) -> ShapeFlow:
        self.layers = self._get_layers(input_shape)
        super().set_input_shape(input_shape)
        C, H, W = input_shape[0]
        if input_shape[1][1] != 1:
            raise ValueError
        self.output_shape = ((self.C, H, W),)
        return self.output_shape

    def get_data(self: DDPMResBlock) -> SaveData:
        data = super().get_data()
        data["C"] = self.C
        return data

    def load_from_data(self: DDPMResBlock, data: SaveData, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data, layer_types)
        self.C = data["C"]
