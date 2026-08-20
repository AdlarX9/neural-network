from __future__ import annotations
from core.exit.exit_loss import ExitLoss
from core.layer.layer import Layer
from .network import Network
from ..layer.embedding import Embedding
from graphics import ConsoleVisualization
from ..tokenizer.byte_tokenizer import ByteTokenizer
from ..utils.typing import Shape, Tensor, TensorFlow, Tokens, TrainData, SaveData


class TextNetwork(Network):
    def __init__(
        self: TextNetwork,
        layers: list[Layer] = [],
        input_shape: Shape = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
        embedding: Embedding = Embedding(),
    ) -> None:
        self.embedding: Embedding = embedding
        super().__init__(layers, input_shape, lr, exit_loss)

    def set_lr(self: TextNetwork, lr: float) -> None:
        super().set_lr(lr)
        self.embedding.set_lr(lr)

    def tokenize(self: TextNetwork, text: str) -> Tokens:
        return self.embedding.tokenize(text)

    def get_one_hot(self: TextNetwork, entry: Tokens) -> Tensor:
        return self.embedding.get_one_hot(entry)

    def get_embedded(self: TextNetwork, entry: Tokens) -> Tensor:
        return self.embedding.get_embedded(entry)

    def get_tokens(self: TextNetwork, entry: Tensor) -> Tokens:
        return self.embedding.get_tokens(entry)

    def untokenize(self: TextNetwork, tokens: Tokens) -> str:
        return self.embedding.untokenize(tokens)

    def backprop(self: TextNetwork, gradient: TensorFlow) -> TensorFlow:
        gradient = super().backprop(gradient)
        return self.embedding.backprop(gradient)

    def compute_text(self: TextNetwork, entry: str, memorize: bool = False) -> str:
        one_hot = self.get_embedded(self.tokenize(entry))
        result = self((one_hot,), memorize)
        output = self.untokenize(self.get_tokens(result[0]))
        return output

    def train_tokens(
        self: TextNetwork,
        data: list[tuple[Tokens, Tokens]],
        batch: int = 100,
        visualization: ConsoleVisualization | None = None,
    ) -> None:
        new_data: TrainData = []
        show = bool(len(data) >= 1000)
        for i in range(len(data)):
            if show:
                progress = i / len(data) * 100
                print("One Hot progress: " f"{progress:.2f}%", end="\r")
            entry = self.get_embedded(data[i][0])
            answer = self.get_one_hot(data[i][1])
            new_data.append((entry, answer))
        if show:
            print("One Hot progress: 100.00%")
        del data
        return super().train(new_data, batch, visualization)

    def print(self: TextNetwork, sentence: str) -> None:
        tokens = self.tokenize(sentence)
        for token in tokens:
            try:
                if isinstance(self.embedding.tokenizer, ByteTokenizer):
                    print(self.embedding.tokenizer._token_bytes[token].decode("utf-8"))
            except:
                print("error")

    def get_data(self: TextNetwork) -> SaveData:
        self.layers.append(self.embedding)
        data = super().get_data()
        self.layers.pop()
        return data

    def load_from_data(self: TextNetwork, data: SaveData, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data, layer_types)
        embedding: Embedding | Layer = self.layers.pop()
        if not isinstance(embedding, Embedding):
            raise MemoryError
        self.embedding = embedding
