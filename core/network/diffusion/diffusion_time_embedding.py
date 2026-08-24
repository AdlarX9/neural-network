from __future__ import annotations
from ...basics.layer import Layer
from ...activation.silu import SiLU
from ...basics.block import Block
from ...block.linear import Linear
from ...transform.sin_embedding import SinEmbedding
from ...utils.typing import Receive1


class DiffusionTimeEmbedding(Block):
    def __init__(
        self: DiffusionTimeEmbedding, dim: tuple[int, int, int] = (512, 2048, 512), receive: Receive1 = (0,)
    ) -> None:
        """
        0: time t (1 x 1) -> | 0: time embedding (512 x 1)
        """
        layers: list[Layer] = [
            SinEmbedding(dim=dim[0]),
            Linear(neuron_number=dim[1]),
            SiLU(),
            Linear(neuron_number=dim[2]),
        ]

        super().__init__(layers, receive)
