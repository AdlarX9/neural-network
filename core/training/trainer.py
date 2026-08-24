from __future__ import annotations
from typing import Any, Callable
import math
import time
import numpy as np
from graphics import ConsoleVisualization
from .data import Data
from ..loss.loss import Loss
from ..loss.squared_loss import SquaredLoss
from ..utils.typing import TensorFlow, ParamGrad, Tensor
from ..basics.layer import Layer


class Trainer:
    def __init__(
        self: Trainer,
        units: list[dict[str, Any]] = [],
        data: Data = Data(),
        loss: Loss = SquaredLoss(),
        checkpoint: Callable[[Trainer, int]] = lambda trainer, epochs: None,
    ) -> None:
        self._storage: dict[Any, Any] = {}
        self.checkpoint: Callable[[Trainer, int]] = checkpoint

        self.t: int = 0
        self.data: Data = data
        self.total_batches: int = 0
        self.total_items: int = 0
        self.started_at: float = 0

        self.loss: Loss = loss
        self.accuracies: list[tuple[float, ...]] = []
        self.losses: list[tuple[float, ...]] = []

        self.layers: list[Layer] = [el["layer"] for el in units]
        for layer in self.layers:
            layer.set_id()

        self.adam: list[bool] = [el.get("adam", False) for el in units]
        self.adamw: list[bool] = [el.get("adamw", False) for el in units]
        self.gradient_clipping: list[bool] = [el.get("gradient_clipping", False) for el in units]

        self.m: list[dict[str, Tensor]] = [{} for _ in range(len(units))]
        self.v: list[dict[str, Tensor]] = [{} for _ in range(len(units))]
        self.beta1: list[float] = [el.get("beta1", 0.9) for el in units]
        self.beta2: list[float] = [el.get("beta2", 0.95) for el in units]
        self.epsilon: list[float] = [el.get("epsilon", 1e-8) for el in units]
        self.weight_decay: list[float] = [el.get("weight_decay", 0) for el in units]
        self.g_max: list[float] = [el.get("g_max", 1) for el in units]

        self.warmup_steps: list[int] = [el.get("warmup_steps", 0) for el in units]
        self.max_lr: list[float] = [units[i].get("max_lr", self.layers[i].lr) for i in range(len(units))]
        self.final_lr: list[float] = [units[i].get("final_lr", self.max_lr[i]) for i in range(len(units))]
        self.cosine_decay: list[tuple[int, int]] = [el.get("cosine_decay", (0, 1)) for el in units]

    def optimize_model(self: Trainer, idx, param_gradients: ParamGrad) -> None:
        """
        On part du principe qu'on entraîne qu'un seul réseau, pour l'instant : à modifier par la suite
        """
        layer: Layer = self.layers[idx]
        max_lr: float = self.max_lr[idx]
        final_lr: float = self.final_lr[idx]
        if self.t <= self.warmup_steps[idx] + 1:
            lr = max_lr / (self.warmup_steps[idx] + 1) * self.t
            layer.set_lr(lr)
        t_0, t_f = self.cosine_decay[idx]
        if t_0 < self.t <= t_f:
            lr = final_lr + (max_lr - final_lr) / 2 * (1 + math.cos(math.pi * (self.t - t_0) / (t_f - t_0)))
            layer.set_lr(lr)
        scale = 1
        if self.gradient_clipping[idx]:
            global_norm = np.sqrt(sum(np.sum(gradient**2) for gradient in param_gradients.values()))
            if global_norm > self.g_max[idx]:
                scale = self.g_max[idx] / global_norm
        for j, grad in param_gradients.items():
            split = j.split(".")
            if len(split) != 2:
                raise ValueError
            layer_id = int(split[0])
            param_name = split[1]
            parameterized = layer.get_layer_by_id(layer_id)
            if parameterized is None:
                raise ValueError
            if parameterized.frozen:
                continue
            param: Tensor = getattr(parameterized, param_name)
            update = grad
            if self.adam[idx] or self.adamw[idx]:
                m: Tensor = self.m[idx].get(j, np.zeros_like(param))
                v: Tensor = self.v[idx].get(j, np.zeros_like(param))
                m = self.beta1[idx] * m + (1 - self.beta1[idx]) * grad
                v = self.beta1[idx] * v + (1 - self.beta1[idx]) * grad**2
                self.m[idx][j] = m
                self.v[idx][j] = v
                m = m / (1 - self.beta1[idx] ** self.t)
                v = v / (1 - self.beta2[idx] ** self.t)
                update = m / (np.sqrt(v) + self.epsilon[idx])
            if self.gradient_clipping[idx]:
                update *= scale
            if self.adamw[idx]:
                new_param = (
                    1 - parameterized.lr * self.weight_decay[idx]
                ) * param - parameterized.lr * update
            else:
                new_param = param - parameterized.lr * update
            setattr(parameterized, param_name, new_param)

    def single_train(self: Trainer, entry: TensorFlow, answer: TensorFlow) -> TensorFlow:
        layer = self.layers[0]
        prediction = layer(entry, memorize=True)
        gradients = self.loss.get_gradient(prediction, answer)
        _, param_gradients = layer.backprop(gradients)
        self.optimize_model(0, param_gradients)
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
            if len(pred.shape) != 2:
                accuracies.append(0.0)
                continue
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
