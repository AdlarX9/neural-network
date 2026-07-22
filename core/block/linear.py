from __future__ import annotations
from .block import Block
from ..layer.fc import FC
from ..layer.biais import Biais


class Linear(Block):
    def __init__(self: Linear, neuron_number: int = 1) -> None:
        super().__init__(FC(neuron_number), Biais())  # Add Identity
