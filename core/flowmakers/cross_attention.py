from __future__ import annotations
import math
from ..basics.block import Block
from ..basics.layer import Layer
from ..parameterized.mhfc import MHFC
from ..parameterized.fc import FC
from ..flowmakers.duplicate import Duplicate
from ..flowmakers.matmul import Matmul
from ..transform.scale import Scale
from ..transform.transpose import Transpose
from ..activation.softmax import Softmax
from ..transform.reshape import Reshape
from ..utils.typing import ShapeFlow, Receive2, SaveData


class CrossAttention(Block):
    def __init__(
        self: CrossAttention, H: int = 1, mask: Layer | None = None, receive: Receive2 = (0, 1)
    ) -> None:
        self.H: int = H
        self.d_h: int = 0
        self.mask = mask
        super().__init__([], receive)

    def _get_layers(self: CrossAttention) -> list[Layer]:
        layers = [
            Duplicate(2, receive=(1,)),
            MHFC(self.H, self.d_h, receive=(0,)),  # Q => flow 0
            MHFC(self.H, self.d_h, receive=(1,)),  # K => flow 1
            MHFC(self.H, self.d_h, receive=(2,)),  # V => flow 1
            Transpose(receive=(1,)),  # K^T
            Matmul(receive=(1, 0)),  # K^T @ Q
            Scale(1 / math.sqrt(self.d_h), receive=(0,)),  # K^T @ Q / sqrt(d_h)
        ]
        if self.mask is not None:
            layers.append(self.mask)
        layers += [
            Softmax(axis=1, receive=(0,)),  # Softmax(K^T @ Q / sqrt(d_h))
            Matmul(receive=(1, 0)),  # V @ Softmax(K^T @ Q / sqrt(d_h))
            Reshape((self.H * self.d_h, -1), receive=(0,)),  # Concat head results
            FC(self.H * self.d_h, receive=(0,)),  # Final projection
        ]
        return layers

    def set_input_shape(self: CrossAttention, input_shape: ShapeFlow) -> ShapeFlow:
        d, _ = input_shape[0]
        if d % self.H != 0:
            raise ValueError("Dimensions mismatch:", d, self.H)
        self.d_h = d // self.H
        self.layers = self._get_layers()
        return super().set_input_shape(input_shape)

    def get_data(self: CrossAttention) -> SaveData:
        data = super().get_data()
        data["H"] = self.H
        data["d_h"] = self.d_h
        return data

    def load_from_data(
        self: CrossAttention,
        data: SaveData,
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        super().load_from_data(data, layer_types)
        self.H = data["H"]
        self.d_h = data["d_h"]
