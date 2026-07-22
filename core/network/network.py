from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..layer.layer import Layer
from ..exit.exit_loss import ExitLoss
from ..block.block import Block
from graphics import ConsoleVisualization


class Network(Block):
    def __init__(
        self: Network,
        layers: list[Layer] = [],
        input_shape: tuple = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        super().__init__(*layers)
        self.set_lr(lr)
        self.set_input_shape(input_shape)
        self.exit_loss: ExitLoss = exit_loss

    def compute(self: Network, entry: NDArray[np.float64], memorize: bool = False) -> NDArray[np.float64]:
        volume = super().compute(entry, memorize)
        volume = self.exit_loss.feed_forward(volume)
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
        prediction = self.compute(entry, memorize=True)
        loss = self.exit_loss.get_loss(prediction, answer)
        gradient = self.exit_loss.get_gradient(prediction, answer)
        correct = self.accuracy(prediction, answer)
        super().backprop(gradient)
        return loss, correct

    def train(
        self: Network,
        data: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
        batch: int = 100,
        visualization: ConsoleVisualization | None = None,
    ) -> None:
        dashboard = visualization if visualization is not None else ConsoleVisualization(batch, len(data))
        try:
            max_remember = 600
            correct_items: list[float] = []
            losses: list[float] = []
            for batch_index in range(1, batch + 1):
                for item_index, el in enumerate(data, start=1):
                    loss, is_correct = self.single_train(el[0], el[1])
                    correct_items.append(is_correct)
                    losses.append(loss)
                    if len(correct_items) > max_remember:
                        correct_items.pop(0)
                    if len(losses) > max_remember:
                        losses.pop(0)
                    accuracy = sum(correct_items) / len(correct_items) if len(correct_items) else 0.0
                    average_loss = sum(losses) / len(losses) if len(losses) else -1.0
                    dashboard.update(batch_index, item_index, average_loss, accuracy)
        finally:
            if visualization is None:
                dashboard.close()

    def get_data(self: Network) -> tuple[list[int], list[float], list[str]]:
        self.layers.append(self.exit_loss)
        int_list, float_list, string_list = super().get_data()
        self.layers.pop()
        return int_list, float_list, string_list

    def load_from_data(
        self: Network,
        int_list: list[int],
        float_list: list[float],
        string_list: list[str],
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        super().load_from_data(int_list, float_list, string_list, layer_types)
        if isinstance(self.layers[-1], ExitLoss):
            self.exit_loss = self.layers[-1]
            self.layers.pop()
        else:
            raise MemoryError("No ExitLoss layer found")
