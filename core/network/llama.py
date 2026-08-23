from __future__ import annotations
from ..text.byte_tokenizer import ByteTokenizer
from ..parameterized.embedding import Embedding
from ..parameterized.norm.rms_norm import RMSNorm
from ..basics.layer import Layer
from ..activation.softmax import Softmax
from ..block.res import Res
from ..basics.block import Block
from ..block.rcmha import RCMHA
from ..block.linear import Linear
from ..block.swiglu import SwiGLU
from ..text.gpt import GPT


class LLaMA(GPT):
    def __init__(
        self: LLaMA,
        vocab_size: int = 8192,
        embedding_dim: int = 128,
        head_numbers: list[int] = [4, 4, 4, 4, 4, 4],
        lr: float = 3e-4,
    ) -> None:
        embedding = Embedding(ByteTokenizer(vocab_size), embedding_dim)
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
                            SwiGLU(int(8 / 3 * embedding_dim)),
                            Linear(embedding_dim),
                        ]
                    )
                )
            )
        layers += [
            RMSNorm(),
            Linear(vocab_size),
            Softmax(axis=0),
        ]
        super().__init__(layers, embedding)
        self.set_lr(lr)
