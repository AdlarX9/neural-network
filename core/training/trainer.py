from __future__ import annotations
from typing import Any, Callable
import math
import time
import numpy as np
from graphics import ConsoleVisualization
from .data import Data
from ..basics.network import Network
from ..loss.loss import Loss
from ..loss.squared_loss import SquaredLoss
from ..utils.typing import TensorFlow, ParamGrad, Tensor


class Trainer:
    def __init__(
        self: Trainer,
        networks: tuple[Network, ...] = (),
        data: Data = Data(),
        loss: Loss = SquaredLoss(),
        checkpoint: Callable[[Trainer, int]] = lambda trainer, epochs: None,
    ) -> None:
        self._storage: dict[Any, Any] = {}
        self.checkpoint: Callable[[Trainer, int]] = checkpoint

        for network in networks:
            network.set_id()
        self.networks: tuple[Network, ...] = networks
        self.loss: Loss = loss

        self.data: Data = data
        self.total_batches: int = 0
        self.total_items: int = 0
        self.started_at: float = 0

        self.accuracies: list[tuple[float, ...]] = []
        self.losses: list[tuple[float, ...]] = []

        self.adam: bool = False
        self.adamw: bool = False
        self.gradient_clipping: bool = False

        self.t: int = 0
        self.m: dict[str, Tensor] = {}
        self.v: dict[str, Tensor] = {}
        self.beta1: float = 0.9
        self.beta2: float = 0.95
        self.epsilon: float = 1e-8
        self.weight_decay: float = 0
        self.g_max: float = 1

        self.warmup_steps: int = 0
        self.max_lr: tuple[float, ...] = tuple([network.lr for network in self.networks])
        self.final_lr: tuple[float, ...] = self.max_lr
        self.cosine_decay: tuple[int, int] = (0, 1)

    def optimize_model(self: Trainer, param_gradients: ParamGrad) -> None:
        """
        On part du principe qu'on entraîne qu'un seul réseau, pour l'instant : à modifier par la suite
        """
        network = self.networks[0]
        max_lr = self.max_lr[0]
        final_lr = self.final_lr[0]
        if self.t <= self.warmup_steps + 1:
            lr = max_lr / (self.warmup_steps + 1) * self.t
            network.set_lr(lr)
        t_0, t_f = self.cosine_decay
        if t_0 < self.t <= t_f:
            lr = final_lr + (max_lr - final_lr) / 2 * (1 + math.cos(math.pi * (self.t - t_0) / (t_f - t_0)))
            network.set_lr(lr)
        scale = 1
        if self.gradient_clipping:
            global_norm = np.sqrt(sum(np.sum(gradient**2) for gradient in param_gradients.values()))
            if global_norm > self.g_max:
                scale = self.g_max / global_norm
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
            update = grad
            if self.adam or self.adamw:
                m: Tensor = self.m.get(idx, np.zeros_like(param))
                v: Tensor = self.v.get(idx, np.zeros_like(param))
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta1 * v + (1 - self.beta1) * grad**2
                self.m[idx] = m
                self.v[idx] = v
                m = m / (1 - self.beta1**self.t)
                v = v / (1 - self.beta2**self.t)
                update = m / (np.sqrt(v) + self.epsilon)
            if self.gradient_clipping:
                update *= scale
            if self.adamw:
                new_param = (1 - layer.lr * self.weight_decay) * param - layer.lr * update
            else:
                new_param = param - layer.lr * update
            setattr(layer, param_name, new_param)

    def single_train(self: Trainer, entry: TensorFlow, answer: TensorFlow) -> TensorFlow:
        network = self.networks[0]
        prediction = network(entry, memorize=True)
        gradients = self.loss.get_gradient(prediction, answer)
        _, param_gradients = network.backprop(gradients)
        self.optimize_model(param_gradients)
        return prediction

    def train(
        self: Trainer,
        batch: int = 100,
        title: str | None = None,
    ) -> None:
        data = self.data.training_set
        self.total_batches = batch
        self.total_items = len(data)
        self.started_at = time.time()

        self.start_visualization(title)

        for batch_index in range(1, batch + 1):
            for item_index in range(1, len(data) + 1):
                self.t += 1
                x = data[item_index - 1][0]
                y = data[item_index - 1][1]
                prediction = self.single_train(x, y)
                self.compute_metrics(prediction, y)
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

    def compute_metrics(self: Trainer, prediction: TensorFlow, answer: TensorFlow) -> None:
        accuracies = self.compute_accuracy(prediction, answer)
        losses = self.loss.get_loss(prediction, answer)
        self.accuracies.append(accuracies)
        self.losses.append(losses)

    def start_visualization(self: Trainer, title: str | None = None) -> None:
        self._storage["dashboard"] = ConsoleVisualization(self)
        if title is not None:
            self._storage["dashboard"].title = title

    def visualize(self: Trainer, batch_index: int, item_index: int) -> None:
        max_remember = 50
        average_loss = [sum(el) / len(el) for el in self.losses[-max_remember:]]
        average_loss = sum(average_loss) / len(average_loss)
        average_accuracy = [sum(el) / len(el) for el in self.accuracies[-max_remember:]]
        average_accuracy = sum(average_accuracy) / len(average_accuracy)
        self._storage["dashboard"].update(batch_index, item_index, average_loss, average_accuracy)

    def end_visualization(self: Trainer) -> None:
        self._storage["dashboard"].close()
