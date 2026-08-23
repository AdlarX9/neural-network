from __future__ import annotations
from .ddpm_pad_mask import DDPMPadMask
from ...basics.layer import Layer
from ...basics.block import Block
from ...transform.reshape import Reshape
from ...utils.typing import Receive1, ShapeFlow, SaveData
from ...block.mha import MHA


class DDPMSelfAttention(Block):
    def __init__(self: DDPMSelfAttention, H: int = 8, receive: Receive1 = (0,)) -> None:
        self.H: int = H
        super().__init__([], receive)

    def _get_layers(self: DDPMSelfAttention, shape: ShapeFlow) -> list[Layer]:
        C, H, W = shape[0]
        d_h = C // H
        layers: list[Layer] = [
            Reshape(shape=(C, H * W)),
            MHA(H, mask=DDPMPadMask()),
            Reshape(shape=(C, H, W)),
        ]
        return layers

    def set_input_shape(self: DDPMSelfAttention, input_shape: ShapeFlow) -> ShapeFlow:
        self.layers = self._get_layers(input_shape)
        return super().set_input_shape(input_shape)

    def get_data(self: DDPMSelfAttention) -> SaveData:
        data = super().get_data()
        data["H"] = self.H
        return data

    def load_from_data(
        self: DDPMSelfAttention,
        data: SaveData,
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        super().load_from_data(data, layer_types)
        self.H = data["H"]
