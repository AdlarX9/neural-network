from __future__ import annotations
import math
from ..basics.block import Block
from ..basics.layer import Layer
from ..parameterized.mhfc import MHFC
from ..parameterized.fc import FC
from ..flowmakers.duplicate import Duplicate
from ..transform.causal import Causal
from ..flowmakers.matmul import Matmul
from ..transform.reshape import Reshape
from ..transform.rope import RoPE
from ..transform.scale import Scale
from ..transform.transpose import Transpose
from ..activation.softmax import Softmax
from ..utils.typing import ShapeFlow, Receive1, SaveData


class RCMHA(Block):
    """Stands for RoPEd Causal Multi-Head Self-Attention"""

    def __init__(self: RCMHA, H: int = 1, receive: Receive1 = (0,)) -> None:
        self.H: int = H
        self.d_h: int = 0
        super().__init__([], receive)

    def _get_layers(self: RCMHA) -> list[Layer]:
        layers = [
            Duplicate(3),
            MHFC(self.H, self.d_h, receive=(0,)),  # Q
            MHFC(self.H, self.d_h, receive=(1,)),  # K
            MHFC(self.H, self.d_h, receive=(2,)),  # V
            RoPE(receive=(0,)),  # RoPE(Q)
            RoPE(receive=(1,)),  # RoPE(K)
            Transpose(receive=(1,)),  # K^T
            Matmul(receive=(1, 0)),  # K^T @ Q
            Scale(1 / math.sqrt(self.d_h), receive=(0,)),  # K^T @ Q / sqrt(d_h)
            Causal(receive=(0,)),  # Causal mask
            Softmax(axis=1, receive=(0,)),  # Softmax(K^T @ Q / sqrt(d_h))
            Matmul(receive=(1, 0)),  # V @ Softmax(K^T @ Q / sqrt(d_h))
            Reshape((self.H * self.d_h, -1), receive=(0,)),  # Concat head results
            FC(self.H * self.d_h, receive=(0,)),  # Final projection
        ]
        return layers

    def set_input_shape(self: RCMHA, input_shape: ShapeFlow) -> ShapeFlow:
        d, _ = input_shape[0]
        if int(d / self.H) != d / self.H:
            raise ValueError("Dimensions mismatch:", d, self.H)
        self.d_h = d // self.H
        self.layers = self._get_layers()
        super().set_input_shape(input_shape)
        self.set_lr(self.lr)
        return self.output_shape

    def get_data(self: RCMHA) -> SaveData:
        data = super().get_data()
        data["H"] = self.H
        data["d_h"] = self.d_h
        return data

    def load_from_data(self: RCMHA, data: SaveData, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data, layer_types)
        self.H = data["H"]
        self.d_h = data["d_h"]
