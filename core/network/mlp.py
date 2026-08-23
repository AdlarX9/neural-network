from __future__ import annotations
from ..parameterized.fc import FC
from ..parameterized.biais import Biais
from ..activation.relu import ReLU
from ..basics.block import Block


class MLP(Block):
    def __init__(
        self: MLP,
        neuron_numbers: list[int] = [1, 1],
        lr: float = 0.001,
    ) -> None:
        if len(neuron_numbers) < 2:
            raise ValueError("Not enough layers:", neuron_numbers)
        input_shape = (neuron_numbers.pop(0), -1)
        layers = []
        for nbr in neuron_numbers:
            layers.append(FC(nbr))
            layers.append(Biais())
            layers.append(ReLU())
        layers.pop()
        super().__init__(layers)
        self.set_lr(lr)
        self.set_input_shape((input_shape,))
