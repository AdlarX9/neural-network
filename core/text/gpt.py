from __future__ import annotations
from core.basics.layer import Layer
from .text_network import TextNetwork
from ..parameterized.embedding import Embedding
from ..utils.typing import Shape


class GPT(TextNetwork):
    def __init__(
        self: GPT,
        layers: list[Layer] = [],
        input_shape: Shape = (0,),
        lr: float = 0.001,
        embedding: Embedding = Embedding(),
    ) -> None:
        super().__init__(layers, input_shape, lr, embedding)

    def generate(self: GPT, sentence: str, nbr_of_tokens: int = 80) -> str:
        for _ in range(nbr_of_tokens):
            new_token = self.predict_next_token(sentence)
            if self.embedding.tokenizer.__class__.__name__ == "WordTokenizer":
                sentence += " "
            sentence += new_token
        return sentence

    def predict_next_token(self: GPT, beginning: str) -> str:
        one_hot = self.get_one_hot(self.tokenize(beginning))
        result = self((one_hot,))
        prediction = self.untokenize(self.get_tokens(result[0])[-1:])
        return prediction
