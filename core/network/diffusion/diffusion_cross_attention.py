from __future__ import annotations
from ...basics.layer import Layer
from ...basics.block import Block
from ...transform.reshape import Reshape
from ...flowmakers.cross_attention import CrossAttention
from ...utils.typing import Receive2, ShapeFlow, SaveData
from .diffusion_pad_mask import DiffusionPadMask


class DiffusionCrossAttention(Block):
    def __init__(self: DiffusionCrossAttention, H: int = 8, receive: Receive2 = (0, 1)) -> None:
        """
        0: image | -> 0: image enrichie
        1: texte |
        """
        self.H = H
        super().__init__([], receive)

    def _get_layers(self: DiffusionCrossAttention, shape: ShapeFlow) -> list[Layer]:
        C, H, W = shape[0]
        layers: list[Layer] = [
            Reshape(shape=(C, H * W), receive=(0,)),
            CrossAttention(H=self.H, mask=DiffusionPadMask(), receive=(0, 1)),
            Reshape(shape=(C, H, W), receive=(0,)),
        ]
        return layers

    def set_input_shape(self: DiffusionCrossAttention, input_shape: ShapeFlow) -> ShapeFlow:
        self.layers = self._get_layers(input_shape)
        return super().set_input_shape(input_shape)

    def get_data(self: DiffusionCrossAttention) -> SaveData:
        data = super().get_data()
        data["H"] = self.H
        return data

    def load_from_data(
        self: DiffusionCrossAttention,
        data: SaveData,
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        super().load_from_data(data, layer_types)
        self.H = data["H"]
