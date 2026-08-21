from __future__ import annotations
from ..parameterized.conv import Conv
from ..parameterized.biais import Biais
from ..activation.relu import ReLU
from ..transform.flatten import Flatten
from .mlp import MLP
from ..basics.layer import Layer
from ..basics.network import Network


class CNN(Network):
    def __init__(
        self: CNN,
        parameters: tuple[list[tuple[int, int, int, int]], list[int]] = ([(1, 1, 1, 1), (1, 1, 1, 1)], [1]),
        lr: float = 0.001,
        more_layers: list[Layer] = []
    ) -> None:
        cnn_params = parameters[0]
        mlp_params = parameters[1]
        if len(cnn_params) < 2:
            raise ValueError("Not enough layers:", parameters)
        input_shape = cnn_params.pop(0)[:3]
        layers: list[Layer] = []
        N, K, S, P = 0, 0, 0, 0
        for cnn_param in cnn_params:
            N, K, S, P = cnn_param
            layers.append(Conv(N, K, S, P))
            layers.append(Biais())
            layers.append(ReLU())
        layers.append(Flatten())
        mlp = MLP(neuron_numbers=[N * K * S] + mlp_params)
        layers.append(mlp)
        layers += more_layers
        super().__init__(layers, input_shape, lr)
