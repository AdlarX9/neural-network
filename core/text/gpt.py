from __future__ import annotations
import numpy as np
from core.basics.layer import Layer
from .text_network import TextNetwork
from ..parameterized.embedding import Embedding
from ..utils.typing import Shape


class GPT(TextNetwork):
    def __init__(
        self: GPT,
        layers: list[Layer] = [],
        embedding: Embedding = Embedding(),
    ) -> None:
        super().__init__(layers, embedding)

    def generate(self: GPT, sentence: str, nbr_of_tokens: int = 80, temperature: float | None = None) -> str:
        for _ in range(nbr_of_tokens):
            new_token = self.predict_next_token(sentence, temperature)
            if self.embedding.tokenizer.__class__.__name__ == "WordTokenizer":
                sentence += " "
            sentence += new_token
        return sentence

    def predict_next_token(self: GPT, beginning: str, temperature: float | None = None) -> str:
        one_hot = self.get_one_hot(self.tokenize(beginning))
        softmax = self.layers.pop()
        logits = self((one_hot,))[0]
        if temperature is not None:
            logits /= temperature
        probs = softmax((logits,))[0]
        self.layers.append(softmax)
        next_token_prob = probs[:, -1].reshape(-1, 1)
        if temperature is not None:
            next_token_index = np.random.choice(len(next_token_prob), p=next_token_prob.ravel())
        else:
            next_token_index = self.get_tokens(next_token_prob)[-1]
        prediction = self.untokenize([next_token_index])
        return prediction
