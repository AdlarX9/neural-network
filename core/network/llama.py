from __future__ import annotations
from ..parameterized.embedding import Embedding
from ..parameterized.rms_norm import RMSNorm
from ..basics.layer import Layer
from ..activation.softmax import Softmax
from ..block.one_hot_maker import OneHotMaker
from ..block.res import Res
from ..basics.block import Block
from ..block.rcmha import RCMHA
from ..block.linear import Linear
from ..block.swiglu import SwiGLU
from ..text.gpt import GPT


class LLaMA(GPT):
    def __init__(
        self: LLaMA,
        head_numbers: list[int] = [],
        embedding: Embedding | None = None,
        lr: float = 0.001,
    ) -> None:
        if embedding is None:
            return
        layers: list[Layer] = []
        for H in head_numbers:
            layers.append(
                Res(
                    Block(
                        [
                            RMSNorm(),
                            RCMHA(H),
                        ]
                    )
                )
            )
            layers.append(
                Res(
                    Block(
                        [
                            RMSNorm(),
                            SwiGLU(int(8 / 3 * embedding.dim)),
                            Linear(embedding.dim),
                        ]
                    )
                )
            )
        layers += [
            RMSNorm(),
            OneHotMaker(embedding),
            Softmax(axis=0),
        ]
        super().__init__(layers, (embedding.dim, -1), lr, embedding)
