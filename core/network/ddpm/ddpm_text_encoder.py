from __future__ import annotations
from .ddpm_pad_mask import DDPMPadMask
from ...basics.layer import Layer
from ...block.res import Res
from ...parameterized.norm.rms_norm import RMSNorm
from ...block.swiglu import SwiGLU
from ...basics.block import Block
from ...block.linear import Linear
from ...text.text_network import TextNetwork
from ...utils.typing import Receive1
from ...parameterized.embedding import Embedding
from ...text.byte_tokenizer import ByteTokenizer
from ...block.mha import MHA
from ...transform.text_pos_embedding import TextPosEmbedding


class DDPMTextEncoder(TextNetwork):
    def __init__(
        self: DDPMTextEncoder,
        vocab_size: int = 8192,
        embedding_dim: int = 128,
        head_numbers=[4, 4, 4, 4],
        receive: Receive1 = (0,),
    ) -> None:
        embedding = Embedding(ByteTokenizer(vocab_size), 128)
        layers: list[Layer] = [TextPosEmbedding()]
        for H in head_numbers:
            layers.append(
                Res(
                    Block(
                        [
                            RMSNorm(),
                            MHA(H, mask=DDPMPadMask()),
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
        ]

        super().__init__(layers, embedding, receive)
