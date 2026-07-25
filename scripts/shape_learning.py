from core import MLP
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

    neuron_numbers = [2, 40, 40, 40, 40, 40, 40, 1]
    mlp = MLP(neuron_numbers, 0.001)
    name = "shape_learning"
    mlp.load(name)

    batch = 600
    data = get_data(batch)
    mlp.train(data=data, batch=batch)
    mlp.save(name)
    regression(mlp, curve)
