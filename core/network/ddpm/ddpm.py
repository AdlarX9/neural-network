from __future__ import annotations

from .ddpm_res_block import DDPMResBlock
from .ddpm_cross_attention import DDPMCrossAttention
from .ddpm_self_attention import DDPMSelfAttention
from .ddpm_up_sample import DDPMUpSample
from .ddpm_down_sample import DDPMDownSample
from .ddpm_time_embedding import DDPMTimeEmbedding

from ...parameterized.norm.group_norm import GroupNorm
from ...parameterized.embedding import Embedding
from ...parameterized.conv import Conv

from ...basics.layer import Layer
from ...basics.network import Network

from ...text.tokenizer import Tokenizer

from ...activation.silu import SiLU


class DDPM(Network):
    def __init__(
        self: DDPM,
        tokenizer: Tokenizer | None = None,
        lr: float = 0.001,
    ) -> None:
        """
        0: image bruitée (3 x 64 x 64) -> | 0: image débruitée (3 x 64 x 64)
        1: texte one hot (V x L)          | 1: time embedding (512 x 1)
        2: time t (1 x 1)                 | 2: text embedding (d x L)
        """
        if tokenizer is None:
            return

        self._receive: int = 3
        self.tokenizer: Tokenizer = tokenizer
        layers: list[Layer] = [
            # Text Embedding
            Embedding(tokenizer=tokenizer, dim=128, receive=(1,)),  # + ajouter vrai text encoder
            # Time Embedding
            DDPMTimeEmbedding(receive=(2,)),
            # DOWN Block 1
            Conv(N=128, K=3, receive=(0,)),  # (128 x 64 x 64)
            DDPMResBlock(receive=(0, 2)),
            DDPMResBlock(receive=(0, 1)),
            # DOWN Block 2
            DDPMDownSample(receive=(0,)),  # (256 x 32 x 32)
            DDPMResBlock(receive=(1, 2)),
            DDPMResBlock(receive=(1, 2)),
            # DOWN Block 3
            DDPMDownSample(receive=(1,)),  # (512 x 16 x 16)
            DDPMResBlock(receive=(2, 3)),
            DDPMSelfAttention(H=8, receive=(2,)),
            DDPMCrossAttention(H=8, receive=(2, 4)),
            DDPMResBlock(receive=(2, 4)),
            # BOTTLENECK
            DDPMDownSample(receive=(2,)),  # (512 x 8 x 8)
            DDPMResBlock(receive=(3, 4)),
            DDPMSelfAttention(H=8, receive=(3,)),
            DDPMCrossAttention(H=8, receive=(3, 5)),
            DDPMResBlock(receive=(3, 5)),
            # UP Block 1
            DDPMUpSample(previous_C=512, receive=(2, 3)),  # (1024 x 16 x 16)
            DDPMResBlock(C=512, receive=(2, 3)),  # (512 x 16 x 16)
            DDPMSelfAttention(H=8, receive=(2,)),
            DDPMCrossAttention(H=8, receive=(2, 4)),
            DDPMResBlock(receive=(2, 4)),
            # UP Block 2
            DDPMUpSample(previous_C=256, receive=(1, 2)),  # (512 x 32 x 32)
            DDPMResBlock(C=256, receive=(1, 2)),  # (256 x 32 x 32)
            DDPMResBlock(receive=(1, 2)),
            # UP Block 3
            DDPMUpSample(previous_C=128, receive=(0, 1)),  # (256 x 64 x 64)
            DDPMResBlock(C=128, receive=(0, 1)),  # (128 x 64 x 64)
            DDPMResBlock(receive=(0, 1)),
            #
            # Head
            GroupNorm(groups=32, receive=(0,)),
            SiLU(receive=(0,)),
            Conv(N=3, K=3, receive=(0,)),  # (3 x 64 x 64)
        ]

        super().__init__(layers, (), lr, receive=(0, 1, 2))
