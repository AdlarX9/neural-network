from __future__ import annotations
import numpy as np
from ...basics.layer import Layer
from ...utils.typing import Receive1, Tensor


class DDPMPadMask(Layer):
    def __init__(self: DDPMPadMask, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.nbr_of_pad: int = 0

    def feed_forward(self: DDPMPadMask, entry: Tensor) -> Tensor:
        n, p = entry.shape
        mask = np.zeros((n, p))
        mask[-self.nbr_of_pad :, :] = -np.inf
        return entry + mask
