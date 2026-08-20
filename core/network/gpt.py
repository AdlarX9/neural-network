from __future__ import annotations
import random
from core.exit.exit_loss import ExitLoss
from core.layer.layer import Layer
from .text_network import TextNetwork
from ..layer.embedding import Embedding
from ..tokenizer.word_tokenizer import WordTokenizer
from ..utils.typing import Shape, Tokens


class GPT(TextNetwork):
    def __init__(
        self: GPT,
        layers: list[Layer] = [],
        input_shape: Shape = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
        embedding: Embedding = Embedding(),
    ) -> None:
        super().__init__(layers, input_shape, lr, exit_loss, embedding)

    def build_data(
        self: GPT, text: str, context_length: int = 256, stride: int = 128
    ) -> list[tuple[Tokens, Tokens]]:
        tokens = self.tokenize(text)
        data: list[tuple[Tokens, Tokens]] = []
        for i in range(0, len(tokens) - context_length, stride):
            data.append((tokens[i : i + context_length], tokens[i + 1 : i + context_length + 1]))
        random.shuffle(data)
        return data

    def generate(self: GPT, sentence: str, nbr_of_tokens: int = 80) -> str:
        for _ in range(nbr_of_tokens):
            new_token = self.predict_next_token(sentence)
            if self.embedding.tokenizer.__class__.__name__ == "WordTokenizer":
                sentence += " "
            sentence += new_token
        return sentence

    def predict_next_token(self: GPT, beginning: str) -> str:
        one_hot = self.get_embedded(self.tokenize(beginning))
        result = self((one_hot,))
        prediction = self.untokenize(self.get_tokens(result[0])[-1:])
        return prediction
