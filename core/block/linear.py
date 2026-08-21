from __future__ import annotations
from ..basics.block import Block
from ..parameterized.fc import FC
from ..parameterized.biais import Biais
from ..utils.typing import Receive1


class Linear(Block):
    def __init__(self: Linear, neuron_number: int = 1, receive: Receive1 = (0,)) -> None:
        super().__init__([FC(neuron_number), Biais()], receive)  # Add Identity
