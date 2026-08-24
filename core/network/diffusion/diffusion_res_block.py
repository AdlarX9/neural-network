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


class DiffusionResBlock(Block):
    def __init__(
        self: DiffusionResBlock, C: int = -1, groups: int = 32, dropout: float = 0.1, receive: Receive2 = (0, 1)
    ) -> None:
        """
        0: image          | -> 0: image transformée
        1: time embedding |
        """
        self.C = C
        self.groups = groups
        self.dropout = dropout
        super().__init__([], receive)

    def _get_layers(self: DiffusionResBlock, shape: ShapeFlow) -> list[Layer]:
        C = shape[0][0]
        if self.C == -1:
            self.C = C
        layers: list[Layer] = [
            Duplicate(factor=2, receive=(0,)),
            GroupNorm(groups=self.groups, receive=(0,)),
            SiLU(receive=(0,)),
            Conv(N=self.C, K=3, receive=(0,)),
            Linear(neuron_number=self.C, receive=(2,)),
            Reshape(shape=(self.C, 1, 1), receive=(2,)),
            Add(receive=(0, 2)),
            GroupNorm(groups=self.groups, receive=(0,)),
            SiLU(receive=(0,)),
            Dropout(p=self.dropout, inverted=True, receive=(0,)),
            Conv(N=self.C, K=3, receive=(0,)),
        ]
        if self.C != C:
            layers += [
                Conv(N=self.C, K=1, receive=(1,)),
            ]
        layers += [
            Add(receive=(0, 1)),
        ]
        return layers

    def set_input_shape(self: DiffusionResBlock, input_shape: ShapeFlow) -> ShapeFlow:
        self.layers = self._get_layers(input_shape)
        super().set_input_shape(input_shape)
        C, H, W = input_shape[0]
        if input_shape[1][1] != 1:
            raise ValueError
        self.output_shape = ((self.C, H, W),)
        return self.output_shape

    def get_data(self: DiffusionResBlock) -> SaveData:
        data = super().get_data()
        data["C"] = self.C
        data["groups"] = self.groups
        data["dropout"] = self.dropout
        return data

    def load_from_data(self: DiffusionResBlock, data: SaveData, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data, layer_types)
        self.C = data["C"]
        self.groups = data["groups"]
        self.dropout = data["dropout"]
