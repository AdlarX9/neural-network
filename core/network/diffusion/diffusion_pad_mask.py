from __future__ import annotations
import numpy as np
from ...basics.layer import Layer
from ...utils.typing import Receive1, Tensor


class DiffusionPadMask(Layer):
    def __init__(self: DiffusionPadMask, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.nbr_of_pad: int = 0

    def feed_forward(self: DiffusionPadMask, entry: Tensor) -> Tensor:
        if self.nbr_of_pad == 0:
            return entry
        C, n, p = entry.shape
        mask = np.zeros((C, n, p))
        mask[:, -self.nbr_of_pad :, :] = -np.inf
        return entry + mask
