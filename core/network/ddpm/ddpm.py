from __future__ import annotations
import numpy as np
from math import cos, sqrt, pi
from .ddpm_text_encoder import DDPMTextEncoder
from .ddpm_pad_mask import DDPMPadMask
from ...basics.layer import Layer
from ...basics.block import Block
from ...block.u_net import UNet
from ...utils.typing import Tensor, TensorFlow, ParamGrad, SaveData


class DDPM(Block):
    def __init__(
        self: DDPM,
        embeddings: list[Layer] = [],
        down_blocks: list[Layer] = [],
        up_blocks: list[Layer] = [],
        head: list[Layer] = [],
        L: int = 64,
        T: int = 1000,
    ) -> None:
        """
        0: image bruitée (1 x H x W) -> 0: bruit prédit (1 x H x W)
        1: texte one hot (V x L)
        2: time t (1 x 1)
        """
        self.output: TensorFlow | None = None
        self.L: int = L
        self.T: int = T
        self.beta_min: float = 1e-4
        self.beta_max: float = 0.02

        u_net = [UNet(down_blocks, up_blocks, receive=(0, 1, 2))]
        layers: list[Layer] = embeddings + u_net + head
        super().__init__(layers, receive=(0, 1, 2))

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
        return self.beta_min + (t-1) / (self.T-1) * (self.beta_max - self.beta_min)

    def get_alpha(self: DDPM, t: int) -> float:
        return 1.0 - self.get_beta(t)

    def get_alpha_bar(self: DDPM, t: int) -> float:
        if not 0 <= t <= self.T:
            raise ValueError(f"t must be in [0, {self.T}], got {t}")
        alpha_bar = 1.0
        for i in range(1, t + 1):
            alpha_bar *= self.get_alpha(i)
        return alpha_bar

    def generate(self: DDPM, prompt: str) -> Tensor:
        text_encoder: DDPMTextEncoder = self[DDPMTextEncoder][0]  # type: ignore
        one_hot = text_encoder.get_one_hot(text_encoder.tokenize(prompt))

        x = np.random.randn(*self.input_shape[0])
        for t in reversed(range(1, self.T + 1)):
            progress = 100 * (self.T - 1 - t) / (self.T - 1)
            print("Generation progress: " f"{progress:.2f}%", end="\r")

            beta = self.get_beta(t)
            alpha = self.get_alpha(t)
            alpha_bar = self.get_alpha_bar(t)
            predicted_noise = self((x, one_hot, np.array([[t]])))[0]
            mu = 1 / sqrt(alpha) * (x - beta / sqrt(1 - alpha_bar) * predicted_noise)
            sigma = sqrt(beta * (1 - self.get_alpha_bar(t - 1)) / (1 - alpha_bar))
            z = np.random.randn(*x.shape)
            x = mu + sigma * z
            if t == 1:
                x = mu  # image finale

        print("Generation progress: 100.00%")
        return x
