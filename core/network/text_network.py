from __future__ import annotations
from core.exit.exit_loss import ExitLoss
from core.layer.layer import Layer
from .network import Network
from ..layer.embedding import Embedding
import numpy as np
from numpy.typing import NDArray
from graphics import ConsoleVisualization


class TextNetwork(Network):
    def __init__(
        self: TextNetwork,
        layers: list[Layer] = [],
        input_shape: tuple = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
        embedding: Embedding = Embedding(),
    ) -> None:
        super().__init__(layers, input_shape, lr, exit_loss)
        self.embedding: Embedding = embedding

    def compute(self: TextNetwork, entry: NDArray[np.float64], memorize: bool = False) -> NDArray[np.float64]:
        embedded = self.embedding.compute(entry, memorize)
        return super().compute(embedded, memorize)

    def tokenize(self: TextNetwork, text: str) -> list[int]:
        return self.embedding.tokenize(text)

    def get_one_hot(self: TextNetwork, entry: list[int]) -> NDArray[np.float64]:
        return self.embedding.get_one_hot(entry)

    def get_tokens(self: TextNetwork, entry: NDArray[np.float64]) -> list[int]:
        return self.embedding.get_tokens(entry)

    def untokenize(self: TextNetwork, tokens: list[int]) -> str:
        return self.embedding.untokenize(tokens)

    def compute_text(self: TextNetwork, entry: str, memorize: bool = False) -> str:
        one_hot = self.get_one_hot(self.tokenize(entry))
        result = self.compute(one_hot, memorize)
        output = self.untokenize(self.get_tokens(result))
        return output

    def train_tokens(
        self: TextNetwork,
        data: list[tuple[list[int], list[int]]],
        batch: int = 100,
        visualization: ConsoleVisualization | None = None,
    ) -> None:
        new_data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for i in range(len(data)):
            entry = self.get_one_hot(data[i][0])
            answer = self.get_one_hot(data[i][1])
            new_data.append((entry, answer))
        return super().train(new_data, batch, visualization)

    def get_data(self: TextNetwork) -> tuple[list[int], list[float], list[str]]:
        self.layers.append(self.embedding)
        lists = super().get_data()
        self.layers.pop()
        return lists

    def load_from_data(
        self: TextNetwork,
        int_list: list[int],
        float_list: list[float],
        string_list: list[str],
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        super().load_from_data(int_list, float_list, string_list, layer_types)
        embedding: Embedding | Layer = self.layers.pop()
        if not isinstance(embedding, Embedding):
            raise MemoryError
        self.embedding = embedding
