from __future__ import annotations
import numpy as np
from .layer import Layer
from .block import Block
from ..utils.typing import Shape, ShapeFlow, Tensor, TensorFlow, TrainData, SaveData, Receive1
from graphics import ConsoleVisualization
import time


class Network(Block):
    def __init__(
        self: Network,
        layers: list[Layer] = [],
        input_shape: Shape = (0,),
        lr: float = 0.001,
        receive: Receive1 = (0,),
    ) -> None:
        super().__init__(layers, receive)
        self.set_lr(lr)
        self.set_input_shape((input_shape,))
