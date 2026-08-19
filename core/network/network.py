from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer
from ..exit.exit_loss import ExitLoss
from ..block.block import Block
from graphics import ConsoleVisualization
import time


class Network(Block):
    def __init__(
        self: Network,
        layers: list[Layer] = [],
        input_shape: tuple[int, ...] = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
        receive: tuple[int] = (0,),
    ) -> None:
        super().__init__(layers, receive)
        self.set_lr(lr)
        self.exit_loss: ExitLoss = exit_loss
        self.set_input_shape((input_shape,))

    def set_input_shape(self: Network, input_shape: tuple) -> tuple:
        output_shape = super().set_input_shape(input_shape)
        self.output_shape = self.exit_loss.set_input_shape(output_shape)
        return self.output_shape

    def __call__(self: Network, entry: tuple, memorize: bool = False) -> tuple:
        volume = super().__call__(entry, memorize)
        volume = self.exit_loss(volume, memorize)
        return volume

    def accuracy(self: Network, prediction: NDArray[np.float64], answer: NDArray[np.float64]) -> float:
        is_correct = []
        _, p = prediction.shape
        for i in range(p):
            is_correct.append(bool(np.argmax(prediction[:, i]) == np.argmax(answer[:, i])))
        return is_correct.count(True) / len(is_correct)

    def single_train(
        self: Network, entry: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> tuple[float, float]:
        """
        On assume qu'on est face à un cas classique : une seule entrée => une seule sortie
        On refera ces algorithmes dans une prochaine issue
        """
        prediction = self((entry,), memorize=True)[0]  # On passe aux tuples pour les entrées
        loss = self.exit_loss.get_loss(prediction, answer)
        gradient = self.exit_loss.get_gradient(prediction, answer)
        correct = self.accuracy(prediction, answer)
        super().backprop((gradient,))  # On passe aux tuples pour les gradients
        return loss, correct

    def train(
        self: Network,
        data: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
        batch: int = 100,
        visualization: ConsoleVisualization | None = None,
    ) -> None:
        """
        On assume qu'on est face à un cas classique : une seule entrée => une seule sortie
        On refera ces algorithmes dans une prochaine issue
        """
        dashboard = visualization if visualization is not None else ConsoleVisualization(batch, len(data))
        dashboard.total_batches = batch
        dashboard.total_items = len(data)
        timestamp = time.time()
        try:
            max_remember = 100
            correct_items: list[float] = []
            losses: list[float] = []
            for batch_index in range(1, batch + 1):
                for item_index in range(1, len(data) + 1):
                    loss, is_correct = self.single_train(data[item_index - 1][0], data[item_index - 1][1])
                    correct_items.append(is_correct)
                    losses.append(loss)
                    if len(correct_items) > max_remember:
                        correct_items.pop(0)
                    if len(losses) > max_remember:
                        losses.pop(0)
                    if item_index % 1000 == 0 and time.time() - timestamp >= 60:
                        self.save("temp_" + self.__class__.__name__)
                        timestamp = time.time()
                    accuracy = sum(correct_items) / len(correct_items) if len(correct_items) else 0.0
                    average_loss = sum(losses) / len(losses) if len(losses) else -1.0
                    dashboard.update(batch_index, item_index, average_loss, accuracy)
        finally:
            if visualization is None:
                dashboard.close()

    def get_data(self: Network) -> dict:
        self.layers.append(self.exit_loss)
        data = super().get_data()
        self.layers.pop()
        return data

    def load_from_data(self: Network, data: dict, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data, layer_types)
        if isinstance(self.layers[-1], ExitLoss):
            self.exit_loss = self.layers[-1]
            self.layers.pop()
        else:
            raise MemoryError("No ExitLoss layer found")
