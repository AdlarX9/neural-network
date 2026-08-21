from __future__ import annotations
from typing import Any, Callable
import numpy as np
import time
from graphics import ConsoleVisualization
from .data import Data
from ..basics.network import Network
from ..loss.loss import Loss
from ..loss.squared_loss import SquaredLoss
from ..utils.typing import TensorFlow, ParamGrad, Tensor


class Trainer:
    def __init__(
        self: Trainer,
        data: Data = Data(),
        checkpoint: Callable[[Trainer, int]] = lambda trainer, epochs: None,
    ) -> None:
        self._storage: dict[Any, Any] = {}
        self.checkpoint: Callable[[Trainer, int]] = checkpoint

        self.data: Data = data
        self.total_batches: int = 0
        self.total_items: int = 0
        self.started_at: float = 0

        self.accuracy: list[tuple[float, ...]] = []
        self.loss: list[tuple[float, ...]] = []

        self.adam: bool = True
        self.warmup_steps: int = 0
        self.scheduler_steps: tuple[int, int] = (0, 0)

    def optimize_model(self: Trainer, network: Network, param_gradients: ParamGrad) -> None:
        for idx, grad in param_gradients.items():
            split = idx.split(".")
            if len(split) != 2:
                raise ValueError
            layer_id = int(split[0])
            param_name = split[1]
            layer = network.get_layer_by_id(layer_id)
            if layer is None:
                raise ValueError
            param: Tensor = getattr(layer, param_name)
            new_param = param - layer.lr * grad  # SGD
            setattr(layer, param_name, new_param)

    def single_train(
        self: Trainer,
        networks: tuple[Network, ...],
        loss: Loss,
        entry: TensorFlow,
        answer: TensorFlow,
    ) -> TensorFlow:
        network = networks[0]
        prediction = network(entry, memorize=True)
        gradients = loss.get_gradient(prediction, answer)
        _, param_gradients = network.backprop(gradients)
        self.optimize_model(network, param_gradients)
        return prediction

    def train(
        self: Trainer,
        networks: tuple[Network, ...],
        loss: Loss = SquaredLoss(),
        batch: int = 100,
        title: str | None = None,
    ) -> None:
        for network in networks:
            network.set_id(0)

        data = self.data.training_set
        self.total_batches = batch
        self.total_items = len(data)
        self.started_at = time.time()
        self.start_visualization(title)

        for batch_index in range(1, batch + 1):
            for item_index in range(1, len(data) + 1):
                x = data[item_index - 1][0]
                y = data[item_index - 1][1]
                prediction = self.single_train(networks, loss, x, y)
                self.compute_metrics(loss, prediction, y)
                self.visualize(batch_index, item_index)
                self.checkpoint(self, batch_index * len(data) + item_index)

        self.end_visualization()

    def compute_accuracy(self: Trainer, prediction: TensorFlow, answer: TensorFlow) -> tuple[float, ...]:
        accuracies: list[float] = []
        for pred, ans in zip(prediction, answer):
            _, p = pred.shape
            if p == 0:
                accuracies.append(0.0)
                continue
            is_correct = []
            for i in range(p):
                is_correct.append(bool(np.argmax(pred[:, i]) == np.argmax(ans[:, i])))
            accuracies.append(is_correct.count(True) / len(is_correct))
        return tuple(accuracies)

    def compute_metrics(self: Trainer, loss: Loss, prediction: TensorFlow, answer: TensorFlow) -> None:
        accuracies = self.compute_accuracy(prediction, answer)
        losses = loss.get_loss(prediction, answer)
        self.accuracy.append(accuracies)
        self.loss.append(losses)

    def start_visualization(self: Trainer, title: str | None = None) -> None:
        self._storage["dashboard"] = ConsoleVisualization(
            self.total_batches,
            self.total_items,
        )
        if title is not None:
            self._storage["dashboard"].title = title

    def visualize(self: Trainer, batch_index: int, item_index: int) -> None:
        max_remember = 500
        average_loss = [sum(el) / len(el) for el in self.loss[-max_remember:]]
        average_loss = sum(average_loss) / len(average_loss)
        average_accuracy = [sum(el) / len(el) for el in self.accuracy[-max_remember:]]
        average_accuracy = sum(average_accuracy) / len(average_accuracy)
        self._storage["dashboard"].update(batch_index, item_index, average_loss, average_accuracy)

    def end_visualization(self: Trainer) -> None:
        self._storage["dashboard"].close()
