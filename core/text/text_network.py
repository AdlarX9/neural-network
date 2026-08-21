from __future__ import annotations
from typing import Any
from core.basics.block import Block
from core.basics.layer import Layer
from ..basics.network import Network
from ..parameterized.embedding import Embedding
from .byte_tokenizer import ByteTokenizer
from ..utils.typing import Shape, Tensor, TensorFlow, Tokens, SaveData, ParamGrad


class TextNetwork(Network):
    def __init__(
        self: TextNetwork,
        layers: list[Layer] = [],
        input_shape: Shape = (0,),
        lr: float = 0.001,
        embedding: Embedding = Embedding(),
    ) -> None:
        self.embedding: Embedding = embedding
        super().__init__(layers, input_shape, lr)

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

    def __call__(self: TextNetwork, entry: TensorFlow, memorize: bool = False) -> TensorFlow:
        entry = self.embedding(entry, memorize)
        return super().__call__(entry, memorize)

    def backprop(self: TextNetwork, gradient: TensorFlow) -> tuple[TensorFlow, ParamGrad]:
        gradient1, params = super().backprop(gradient)
        gradient2, params2 = self.embedding.backprop(gradient1)
        params |= params2
        return gradient2, params

    def compute_text(self: TextNetwork, entry: str, memorize: bool = False) -> str:
        one_hot = self.get_one_hot(self.tokenize(entry))
        result = self((one_hot,), memorize)
        output = self.untokenize(self.get_tokens(result[0]))
        return output

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

    def get_params_data(self: TextNetwork) -> list[dict[str, Any]]:
        return self.embedding.get_params_data() + super().get_params_data()

    def set_id(self: TextNetwork, id: int = 0) -> int:
        if self._id != -1:
            return 0
        id = self.embedding.set_id(id)
        return super().set_id(id)

    def get_layer_by_id(self: TextNetwork, id: int) -> Layer | None:
        if self.embedding._id == id:
            return self.embedding
        return super().get_layer_by_id(id)
