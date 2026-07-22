from __future__ import annotations
from core.exit.exit_loss import ExitLoss
from core.layer.layer import Layer
from ..utils.word_manipulator import WordManipulator
from .network import Network
import numpy as np
from numpy.typing import NDArray
from graphics import ConsoleVisualization


class WordNetwork(Network, WordManipulator):
    def __init__(
        self: WordNetwork,
        layers: list[Layer] = [],
        input_shape: tuple = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        WordManipulator.__init__(self)
        Network.__init__(self, layers, input_shape, lr, exit_loss)

    def compute_words(self: WordNetwork, entry: list[str], memorize: bool = False) -> list[str]:
        one_hot = self.get_one_hot(entry)
        result = super().compute(one_hot, memorize)
        words = self.get_words(result)
        return words

    def train_words(
        self: WordNetwork,
        data: list[tuple[list[str], list[str]]],
        batch: int = 100,
        visualization: ConsoleVisualization | None = None,
    ) -> None:
        new_data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for i in range(len(data)):
            entry = self.get_one_hot(data[i][0])
            answer = self.get_one_hot(data[i][1])
            new_data.append((entry, answer))
        return super().train(new_data, batch, visualization)

    def load_from_data(
        self: WordNetwork,
        int_list: list[int],
        float_list: list[float],
        string_list: list[str],
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        WordManipulator.__init__(self)
        Network.load_from_data(self, int_list, float_list, string_list, layer_types)
