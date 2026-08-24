from __future__ import annotations

from .diffusion import Diffusion
from .diffusion_res_block import DiffusionResBlock
from .diffusion_cross_attention import DiffusionCrossAttention
from .diffusion_self_attention import DiffusionSelfAttention
from .diffusion_up_sample import DiffusionUpSample
from .diffusion_time_embedding import DiffusionTimeEmbedding
from .diffusion_text_encoder import DiffusionTextEncoder

from ...parameterized.norm.group_norm import GroupNorm
from ...parameterized.conv import Conv

from ...basics.layer import Layer
from ...basics.block import Block

from ...activation.silu import SiLU
from ...flowmakers.duplicate import Duplicate
from ...text.tokenizer import Tokenizer
from ...utils.typing import ShapeFlow


class Diffusion50M(Diffusion):
    def __init__(
        self: Diffusion,
        L: int = 64,
        T: int = 1000,
        head_numbers: list[int] = [4, 4, 4, 4],
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if tokenizer is None:
            return
        levels: list[int] = [1, 128, 256, 512, 512]

        embeddings: list[Layer] = [
            # Text Embedding
            DiffusionTextEncoder(tokenizer, 128, head_numbers, receive=(1,)),
            # Time Embedding
            DiffusionTimeEmbedding(dim=(512, 2048, 512), receive=(2,)),
            # Image Embedding
            Conv(N=levels[1], K=3, receive=(0,)),  # (128 x H x W)
            Duplicate(factor=3, receive=(2,)),
            DiffusionResBlock(receive=(0, 2)),
            DiffusionResBlock(receive=(0, 2)),
        ]

        down_blocks: list[Layer] = [
            Block(  # DOWN BLOCK 1
                receive=(0, 2),
                layers=[
                    Conv(N=levels[2], K=3, S=2, P=1),  # (256 x H/2 x W/2)
                    Duplicate(factor=2, receive=(1,)),
                    DiffusionResBlock(receive=(0, 1)),
                    DiffusionResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # DOWN BLOCK 2
                receive=(0, 1, 2),
                layers=[
                    Conv(N=levels[3], K=3, S=2, P=1),  # (512 x H/4 x W/4)
                    Duplicate(factor=2, receive=(2,)),
                    DiffusionResBlock(receive=(0, 2)),
                    DiffusionSelfAttention(H=8, receive=(0,)),
                    DiffusionCrossAttention(H=8, receive=(0, 1)),
                    DiffusionResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # DOWN BLOCK 3 (BOTTLENECK)
                receive=(0, 1, 2),
                layers=[
                    Conv(N=levels[4], K=3, S=2, P=1),  # (512 x H/8 x W/8)
                    Duplicate(factor=2, receive=(2,)),
                    DiffusionResBlock(receive=(0, 2)),
                    DiffusionSelfAttention(H=8, receive=(0,)),
                    DiffusionCrossAttention(H=8, receive=(0, 1)),
                    DiffusionResBlock(receive=(0, 1)),
                ],
            ),
        ]

        up_blocks: list[Layer] = [
            Block(  # UP BLOCK 1
                receive=(0, 1, 2),
                layers=[
                    DiffusionUpSample(previous_C=levels[3], receive=(0, 1)),  # (1024 x H/4 x W/4)
                    Duplicate(factor=2, receive=(2,)),
                    DiffusionResBlock(C=levels[3], receive=(0, 2)),  # (512 x H/4 x W/4)
                    DiffusionSelfAttention(H=8, receive=(0,)),
                    DiffusionCrossAttention(H=8, receive=(0, 1)),
                    DiffusionResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # UP BLOCK 2
                receive=(0, 2),
                layers=[
                    DiffusionUpSample(previous_C=levels[2], receive=(0, 1)),  # (512 x H/2 x W/2)
                    Duplicate(factor=2, receive=(1,)),
                    DiffusionResBlock(C=levels[2], receive=(0, 1)),  # (256 x H/2 x W/2)
                    DiffusionResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # UP BLOCK 3
                receive=(0, 2),
                layers=[
                    DiffusionUpSample(previous_C=levels[1], receive=(0, 1)),  # (256 x H x W)
                    Duplicate(factor=2, receive=(1,)),
                    DiffusionResBlock(C=levels[1], receive=(0, 1)),  # (128 x H x W)
                    DiffusionResBlock(receive=(0, 1)),
                ],
            ),
        ]

        head: list[Layer] = [
            GroupNorm(groups=32, receive=(0,)),
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
