from core import MLP, Data, Trainer, TrainData
from graphics import regression
import numpy as np
import math
import random


def shape_learning() -> None:
    def curve(x: float, y: float) -> float:
        circle = math.sqrt(0.5 * (x - 0.5) ** 2 + (y - 0.5) ** 2)
        donut = math.sin(7.5 * circle)
        return donut

    def get_data(dim: int) -> Data:
        data: TrainData = []
        for _ in range(dim):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            data.append(((np.array([[x], [y]]),), (np.array([[curve(x, y)]]),)))
        return Data(data)

    lr = 0.001
    neuron_numbers = [2, 40, 40, 40, 40, 40, 40, 1]
    mlp = MLP(neuron_numbers.copy(), lr)
    name = "shape_learning"
    if mlp.load(name):
        try:
            mlp((np.array([[0.5], [0.5]]),), memorize=False)
        except Exception:
            mlp = MLP(neuron_numbers.copy(), lr)

    batch = 400
    units = [
        {
            "layer": mlp,
        }
    ]
    trainer = Trainer(units, get_data(batch))
    trainer.train(batch=batch)
    mlp.save(name)
    regression(mlp, curve)
