from __future__ import annotations
import numpy as np
from math import cos, sqrt, pi

from .ddpm_res_block import DDPMResBlock
from .ddpm_cross_attention import DDPMCrossAttention
from .ddpm_self_attention import DDPMSelfAttention
from .ddpm_up_sample import DDPMUpSample
from .ddpm_time_embedding import DDPMTimeEmbedding
from .ddpm_text_encoder import DDPMTextEncoder
from .ddpm_pad_mask import DDPMPadMask

from ...parameterized.norm.group_norm import GroupNorm
from ...parameterized.conv import Conv

from ...basics.layer import Layer
from ...basics.block import Block

from ...activation.silu import SiLU
from ...block.u_net import UNet
from ...flowmakers.duplicate import Duplicate
from ...text.tokenizer import Tokenizer
from ...utils.typing import Tensor, TensorFlow, ShapeFlow, ParamGrad, SaveData


class DDPM(Block):
    def __init__(
        self: DDPM,
        L: int = 64,
        T: int = 1000,
        head_numbers: list[int] = [4, 4, 4, 4],
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """
        0: image bruitée (1 x H x W) -> 0: bruit prédit (1 x H x W)
        1: texte one hot (V x L)
        2: time t (1 x 1)
        """
        self.output: TensorFlow | None = None
        if tokenizer is None:
            return

        self.L: int = L
        self.T: int = T
        levels: list[int] = [1, 128, 256, 512, 512]
        input_shape: ShapeFlow = (
            (levels[0], 32, 32),
            (tokenizer.length(), self.L),
            (1, 1),
        )
        self.beta_min: float = 1e-4
        self.beta_max: float = 0.02

        embeddings: list[Layer] = [
            # Text Embedding
            DDPMTextEncoder(tokenizer, 128, head_numbers, receive=(1,)),
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
        self.set_id()

    def __call__(self: DDPM, entry: TensorFlow, memorize: bool = False) -> TensorFlow:
        image, text_one_hot, time = entry

        n, p = text_one_hot.shape
        nbr_of_pad = self.L - p
        if nbr_of_pad > 0:
            pad = np.zeros((n, nbr_of_pad))
            text_one_hot = np.hstack((text_one_hot, pad))
            for padmask in self[DDPMPadMask]:  # type: ignore
                padmask.nbr_of_pad = nbr_of_pad
        elif nbr_of_pad < 0:
            raise ValueError("too many tokens")

        output = super().__call__((image, text_one_hot, time), memorize)
        if memorize:
            self.output = output

        return (output[0],)

    def backprop(self: DDPM, gradient: TensorFlow) -> tuple[TensorFlow, ParamGrad]:
        if self.output is None:
            raise MemoryError
        gradient = (gradient[0], np.zeros_like(self.output[1]), np.zeros_like(self.output[2]))
        return super().backprop(gradient)

    def get_data(self: DDPM) -> SaveData:
        data = super().get_data()
        data["L"] = self.L
        data["T"] = self.T
        data["beta_min"] = self.beta_min
        data["beta_max"] = self.beta_max
        return data

    def load_from_data(self: DDPM, data: SaveData, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data, layer_types)
        self.L = data["L"]
        self.T = data["T"]
        self.beta_min = data["beta_min"]
        self.beta_max = data["beta_max"]

    def get_beta(self: DDPM, t: int) -> float:
        # return self.beta_min + t / (self.T - 1) * (self.beta_max - self.beta_min)
        s = 0.008
        T = self.T
        return 1 - cos(pi / 2 * (t / T + s) / (1 + s)) ** 2 / cos(pi / 2 * ((t - 1) / T + s) / (1 + s)) ** 2

    def get_alpha(self: DDPM, t: int) -> float:
        return 1 - self.get_beta(t)

    def get_alpha_bar(self: DDPM, t: int) -> float:
        alpha = 1
        for i in range(1, t + 1):
            alpha *= self.get_alpha(i)
        return alpha

    def generate(self: DDPM, prompt: str) -> Tensor:
        text_encoder: DDPMTextEncoder = self[DDPMTextEncoder][0]  # type: ignore
        one_hot = text_encoder.get_one_hot(text_encoder.tokenize(prompt))

        x = np.random.randn(32, 32)
        for t in reversed(range(self.T)):
            beta = self.get_beta(t)
            alpha = self.get_alpha(t)
            alpha_bar = self.get_alpha_bar(t)

            predicted_noise = self((x, one_hot, np.array([[t]])))[0]
            mu = 1 / sqrt(alpha) * (x - beta / (1 - alpha_bar) * predicted_noise)
            sigma = sqrt(beta * (1 - self.get_alpha_bar(t - 1)) / (1 - alpha_bar))
            z = np.random.randn(32, 32)
            x = mu + sigma * z
            if t == 0:
                x = mu  # image finale
        return x
