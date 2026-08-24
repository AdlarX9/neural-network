from __future__ import annotations

from .ddpm import DDPM
from .ddpm_res_block import DDPMResBlock
from .ddpm_cross_attention import DDPMCrossAttention
from .ddpm_self_attention import DDPMSelfAttention
from .ddpm_up_sample import DDPMUpSample
from .ddpm_time_embedding import DDPMTimeEmbedding
from .ddpm_text_encoder import DDPMTextEncoder

from ...parameterized.norm.group_norm import GroupNorm
from ...parameterized.conv import Conv

from ...basics.layer import Layer
from ...basics.block import Block

from ...activation.silu import SiLU
from ...flowmakers.duplicate import Duplicate
from ...text.tokenizer import Tokenizer
from ...utils.typing import ShapeFlow


class DDPM2M(DDPM):
    def __init__(
        self: DDPM,
        L: int = 16,
        T: int = 200,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if tokenizer is None:
            return
        levels: list[int] = [1, 32, 64, 128]
        groups = 8
        dropout = 0.1

        embeddings: list[Layer] = [
            # Text Embedding
            DDPMTextEncoder(tokenizer, 32, [], receive=(1,)),
            # Time Embedding
            DDPMTimeEmbedding(dim=(128, 256, 128), receive=(2,)),
            # Image Embedding
            Conv(N=levels[1], K=3, receive=(0,)),  # (32 x H x W)
            Duplicate(factor=2, receive=(2,)),
            DDPMResBlock(groups=groups, dropout=dropout, receive=(0, 2)),
        ]

        down_blocks: list[Layer] = [
            Block(  # DOWN BLOCK 1
                receive=(0, 2),
                layers=[
                    Conv(N=levels[2], K=3, S=2, P=1),  # (64 x H/2 x W/2)
                    DDPMResBlock(groups=groups, dropout=dropout, receive=(0, 1)),
                ],
            ),
            Block(  # DOWN BLOCK 2 (BOTTLENECK)
                receive=(0, 1, 2),
                layers=[
                    Conv(N=levels[3], K=3, S=2, P=1),  # (128 x H/4 x W/4)
                    Duplicate(factor=2, receive=(2,)),
                    DDPMResBlock(groups=groups, dropout=dropout, receive=(0, 2)),
                    DDPMSelfAttention(H=2, receive=(0,)),
                    DDPMCrossAttention(H=4, receive=(0, 1)),
                    DDPMResBlock(groups=groups, dropout=dropout, receive=(0, 1)),
                ],
            ),
        ]

        up_blocks: list[Layer] = [
            Block(  # UP BLOCK 1
                receive=(0, 2),
                layers=[
                    DDPMUpSample(previous_C=levels[2], receive=(0, 1)),  # (128 x H/2 x W/2)
                    DDPMResBlock(
                        C=levels[2], groups=groups, dropout=dropout, receive=(0, 1)
                    ),  # (64 x H/2 x W/2)
                ],
            ),
            Block(  # UP BLOCK 2
                receive=(0, 2),
                layers=[
                    DDPMUpSample(previous_C=levels[1], receive=(0, 1)),  # (64 x H x W)
                    DDPMResBlock(C=levels[1], groups=groups, dropout=dropout, receive=(0, 1)),  # (32 x H x W)
                ],
            ),
        ]

        head: list[Layer] = [
            GroupNorm(groups=groups, receive=(0,)),
            SiLU(receive=(0,)),
            Conv(N=levels[0], K=3, receive=(0,)),  # (1 x H x W)
        ]

        super().__init__(embeddings, down_blocks, up_blocks, head, L, T)  # type: ignore

        input_shape: ShapeFlow = (
            (levels[0], 32, 32),
            (tokenizer.length(), self.L),
            (1, 1),
        )
        self.set_lr(1e-3)
        self.set_input_shape(input_shape)
