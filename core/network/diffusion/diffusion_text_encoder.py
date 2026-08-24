from __future__ import annotations
from .diffusion_pad_mask import DiffusionPadMask
from ...basics.layer import Layer
from ...block.res import Res
from ...parameterized.norm.rms_norm import RMSNorm
from ...block.swiglu import SwiGLU
from ...basics.block import Block
from ...block.linear import Linear
from ...text.text_network import TextNetwork
from ...utils.typing import Receive1
from ...parameterized.embedding import Embedding
from ...text.tokenizer import Tokenizer
from ...block.mha import MHA
from ...transform.text_pos_embedding import TextPosEmbedding


class DiffusionTextEncoder(TextNetwork):
    def __init__(
        self: DiffusionTextEncoder,
        tokenizer: Tokenizer | None = None,
        embedding_dim: int = 128,
        head_numbers=[4, 4, 4, 4],
        receive: Receive1 = (0,),
    ) -> None:
        if tokenizer is None:
            return
        embedding = Embedding(tokenizer, 128)
        layers: list[Layer] = [TextPosEmbedding()]
        for H in head_numbers:
            layers.append(
                Res(
                    Block(
                        [
                            RMSNorm(),
                            MHA(H, mask=DiffusionPadMask()),
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
        ]

        super().__init__(
            layers=layers,
            embedding=embedding,
            receive=receive,
        )
