from __future__ import annotations

from .ddpm_res_block import DDPMResBlock
from .ddpm_cross_attention import DDPMCrossAttention
from .ddpm_self_attention import DDPMSelfAttention
from .ddpm_up_sample import DDPMUpSample
from .ddpm_time_embedding import DDPMTimeEmbedding
from .ddpm_text_encoder import DDPMTextEncoder

from ...parameterized.norm.group_norm import GroupNorm
from ...parameterized.embedding import Embedding
from ...parameterized.conv import Conv

from ...basics.layer import Layer
from ...basics.block import Block

from ...text.byte_tokenizer import ByteTokenizer
from ...activation.silu import SiLU
from ...block.u_net import UNet
from ...flowmakers.duplicate import Duplicate


class DDPM(Block):
    def __init__(self: DDPM) -> None:
        """
        0: image bruitée (1 x H x W) -> | 0: image débruitée (1 x H x W)
        1: texte one hot (V x L)        | 1: text embedding (d x L)
        2: time t (1 x 1)               | 2: time embedding (512 x 1)
        """

        self.L = 64
        self.T = 1000
        levels = [1, 128, 256, 512, 512]
        input_shape = (
            (levels[0], 32, 32),
            (16384, -1),
            (1, 1),
        )

        embeddings: list[Layer] = [
            # Text Embedding
            DDPMTextEncoder(input_shape[1][0], 128, [4, 4, 4, 4]),
            # Time Embedding
            DDPMTimeEmbedding(dim=(512, 2048, 512), receive=(2,)),
            # Image Embedding
            Conv(N=levels[1], K=3, receive=(0,)),  # (128 x H x W)
            Duplicate(factor=3, receive=(2,)),
            DDPMResBlock(receive=(0, 2)),
            DDPMResBlock(receive=(0, 2)),
        ]

        down_blocks: list[Layer] = [
            Block(  # DOWN BLOCK 1
                receive=(0, 2),
                layers=[
                    Conv(N=levels[2], K=3, S=2, P=1),  # (256 x H/2 x W/2)
                    Duplicate(factor=2, receive=(1,)),
                    DDPMResBlock(receive=(0, 1)),
                    DDPMResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # DOWN BLOCK 2
                receive=(0, 1, 2),
                layers=[
                    Conv(N=levels[3], K=3, S=2, P=1),  # (512 x H/4 x W/4)
                    Duplicate(factor=2, receive=(2,)),
                    DDPMResBlock(receive=(0, 2)),
                    DDPMSelfAttention(H=8, receive=(0,)),
                    DDPMCrossAttention(H=8, receive=(0, 1)),
                    DDPMResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # DOWN BLOCK 3 (BOTTLENECK)
                receive=(0, 1, 2),
                layers=[
                    Conv(N=levels[4], K=3, S=2, P=1),  # (512 x H/8 x W/8)
                    Duplicate(factor=2, receive=(2,)),
                    DDPMResBlock(receive=(0, 2)),
                    DDPMSelfAttention(H=8, receive=(0,)),
                    DDPMCrossAttention(H=8, receive=(0, 1)),
                    DDPMResBlock(receive=(0, 1)),
                ],
            ),
        ]

        up_blocks: list[Layer] = [
            Block(  # UP BLOCK 1
                receive=(0, 1, 2),
                layers=[
                    DDPMUpSample(previous_C=levels[3], receive=(0, 1)),  # (1024 x H/4 x W/4)
                    Duplicate(factor=2, receive=(2,)),
                    DDPMResBlock(C=levels[3], receive=(0, 2)),  # (512 x H/4 x W/4)
                    DDPMSelfAttention(H=8, receive=(0,)),
                    DDPMCrossAttention(H=8, receive=(0, 1)),
                    DDPMResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # UP BLOCK 2
                receive=(0, 2),
                layers=[
                    DDPMUpSample(previous_C=levels[2], receive=(0, 1)),  # (512 x H/2 x W/2)
                    Duplicate(factor=2, receive=(1,)),
                    DDPMResBlock(C=levels[2], receive=(0, 1)),  # (256 x H/2 x W/2)
                    DDPMResBlock(receive=(0, 1)),
                ],
            ),
            Block(  # UP BLOCK 3
                receive=(0, 2),
                layers=[
                    DDPMUpSample(previous_C=levels[1], receive=(0, 1)),  # (256 x H x W)
                    Duplicate(factor=2, receive=(1,)),
                    DDPMResBlock(C=levels[1], receive=(0, 1)),  # (128 x H x W)
                    DDPMResBlock(receive=(0, 1)),
                ],
            ),
        ]

        head: list[Layer] = [
            GroupNorm(groups=32, receive=(0,)),
            SiLU(receive=(0,)),
            Conv(N=levels[0], K=3, receive=(0,)),  # (1 x H x W)
        ]

        u_net = [UNet(down_blocks, up_blocks, receive=(0, 1, 2))]
        layers: list[Layer] = embeddings + u_net + head
        super().__init__(layers, receive=(0, 1, 2))
        self.set_lr(1e-3)
        self.set_input_shape(input_shape)
