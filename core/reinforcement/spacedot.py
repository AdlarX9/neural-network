from __future__ import annotations
from ..utils.typing import TensorFlow
from .environment import Environment
import numpy as np


class Spacedot(Environment):
    def __init__(self: Spacedot, dimensions: tuple[int, int] = (800, 600)) -> None:
        super().__init__()
        self.framerate = 60
        self.dimensions = dimensions
        self.pos = np.array([[self.dimensions[0] / 2], [self.dimensions[1] / 2]])
        self.speed = np.zeros((2, 1))
        self.state = (self.pos,)

    def update(self: Spacedot, action: TensorFlow) -> None:
        acceleration = action[0]
        self.speed += acceleration * self.framerate
        self.pos += self.speed * self.framerate
