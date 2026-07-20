from core import (
    FC,
    ExitLoss,
    Network,
    Layer,
    Biais,
    ReLU,
)
from data import SaveHandler
from graphics import regression
import numpy as np
from numpy.typing import NDArray
import math
import random


def shape_learning() -> None:
    def curve(x: float, y: float) -> float:
        circle = math.sqrt(0.5 * (x - 0.5) ** 2 + (y - 0.5) ** 2)
        donut = math.sin(7.5 * circle)
        return donut

    def get_data(dim: int) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        data = []
        for _ in range(dim):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            data.append((np.array([[x], [y]]), np.array([[curve(x, y)]])))
        return data

    layers: list[Layer] = [
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(1),
        Biais(),
    ]
    network = Network(layers=layers, exit_loss=ExitLoss(), input_shape=(2, 1), lr=0.0001)

    save_handler = SaveHandler()
    name = "reproduce_shape"
    if save_handler.has(name):
        network = save_handler.load(name)
        if not isinstance(network, Network):
            raise MemoryError

    batch = 1000
    data = get_data(batch)
    network.train(data=data, batch=batch)
    save_handler.save(network, name)
    regression(network, curve)
