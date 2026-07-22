from __future__ import annotations
from ..exit.exit_loss import ExitLoss
from ..layer.conv import Conv
from ..layer.conv_biais import ConvBiais
from ..activation.relu import ReLU
from ..transform.flatten import Flatten
from .mlp import MLP
from .network import Network


class CNN(Network):
    def __init__(
        self: CNN,
        parameters: tuple[list[tuple[int, int, int, int]], list[int]] = ([(1, 1, 1, 1), (1, 1, 1, 1)], [1]),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        cnn_params = parameters[0]
        mlp_params = parameters[1]
        if len(cnn_params) < 2:
            raise ValueError("Not enough layers:", parameters)
        input_shape = cnn_params.pop(0)[:3]
        layers = []
        N, K, S, P = 0, 0, 0, 0
        for cnn_param in cnn_params:
            N, K, S, P = cnn_param
            layers.append(Conv(N, K, S, P))
            layers.append(ConvBiais())
            layers.append(ReLU())
        layers.append(Flatten())
        mlp = MLP(neuron_numbers=[N * K * S] + mlp_params)
        layers.append(mlp)
        super().__init__(layers, input_shape, lr, exit_loss)
