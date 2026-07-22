from __future__ import annotations
from ..exit.exit_loss import ExitLoss
from ..layer.fc import FC
from ..layer.biais import Biais
from ..activation.relu import ReLU
from .network import Network


class MLP(Network):
    def __init__(
        self: MLP,
        neuron_numbers: list[int] = [1, 1],
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
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
        super().__init__(layers, input_shape, lr, exit_loss)
