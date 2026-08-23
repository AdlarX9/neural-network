from __future__ import annotations
import numpy as np
from ..basics.layer import Layer
from ..utils.functions import softmax
from ..text.tokenizer import Tokenizer
from ..text.byte_tokenizer import ByteTokenizer
from ..text.word_tokenizer import WordTokenizer
from ..utils.typing import ShapeFlow, Tensor, Tokens, SaveData, ParamGrad, Receive1
from graphics import ConsoleVisualization
import math


class Embedding(Layer):
    def __init__(
        self: Embedding,
        tokenizer: Tokenizer | None = None,
        dim: int = 100,
        receive: Receive1 = (0,),
    ) -> None:
        Layer.__init__(self, receive)
        self.dim: int = dim
        self.tokenizer = ByteTokenizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.set_input_shape(((self.tokenizer.length(), -1),))
        else:
            return
        self.W: Tensor = np.random.normal(
            -1 / np.sqrt(self.dim), 1 / np.sqrt(self.dim), (self.dim, self.tokenizer.length())
        )
        self.W_prime: Tensor = np.random.normal(
            -1 / np.sqrt(self.dim), 1 / np.sqrt(self.dim), (self.tokenizer.length(), self.dim)
        )
        self.parameters = ["W"]

    def set_input_shape(self: Embedding, input_shape: ShapeFlow) -> ShapeFlow:
        if len(input_shape[0]) != 2 or input_shape[0][0] != self.tokenizer.length():
            raise ValueError(
                "Expected dimension does not fit Tokenizer requiremenents:",
                (self.tokenizer.length(), 1),
                "!=",
                input_shape,
            )
        self.W: Tensor = np.random.normal(
            -1 / np.sqrt(self.dim), 1 / np.sqrt(self.dim), (self.dim, self.tokenizer.length())
        )
        self.W_prime: Tensor = np.random.normal(
            -1 / np.sqrt(self.dim), 1 / np.sqrt(self.dim), (self.tokenizer.length(), self.dim)
        )
        super().set_input_shape(input_shape)
        self.output_shape = ((self.dim, self.input_shape[0][1]),)
        return self.output_shape

    def feed_forward(self: Embedding, entry: Tensor) -> Tensor:
        return self.W @ entry

    def descend_gradient(self: Embedding, gradient: Tensor) -> Tensor:
        if self.input is None:
            raise MemoryError
        return self.W.T @ gradient

    def params_gradient(self: Embedding, gradient) -> ParamGrad:
        if self.input is None:
            raise MemoryError
        return {"W": gradient @ self.input[0].T}

    def cbow_training(self: Embedding, text: str | Tokens, window: int = 7, batch: int = 10) -> None:
        if isinstance(text, list):
            tokens = text
        else:
            tokens: Tokens = self.tokenizer.tokenize(text)
        dashboard = ConsoleVisualization(batch, len(tokens) - 2 * window)
        max_length: int = 500
        corrects: list[bool] = []
        losses: list[float] = []
        for batch_index in range(1, batch + 1):
            for i in range(window, len(tokens) - window):
                token = tokens[i]

                # Context vector
                context_tokens = tokens[i - window : i] + tokens[i + 1 : i + window + 1]
                one_hots = [self.tokenizer.get_one_hot(word) for word in context_tokens]
                embedded_contexts: list[Tensor] = [
                    self.W[:, context_token].reshape(-1, 1) for context_token in context_tokens
                ]
                embedded_context = sum(embedded_contexts) / len(embedded_contexts)

                # Compute score
                score = self.W_prime @ embedded_context
                probability = softmax(score)

                # Learn
                gradient = probability.copy()
                gradient[token, 0] -= 1
                new_gradient = self.W_prime.T @ gradient
                self.W_prime -= self.lr * gradient @ embedded_context.T  # type: ignore
                self.W -= self.lr * new_gradient @ sum(one_hots).T / (2 * window)  # type: ignore

                # Dashboard
                correct = bool(np.argmax(probability) == token)
                corrects.append(correct)
                if len(corrects) > max_length:
                    corrects.pop(0)
                loss = -math.log(probability[token, 0])
                losses.append(loss)
                if len(losses) > max_length:
                    losses.pop(0)
                dashboard.update(
                    batch_index,
                    i + 1 - window,
                    sum(losses) / len(losses),
                    corrects.count(True) / len(corrects),
                )

    def build_vocab(self: Embedding, text: str) -> None:
        self.tokenizer.build_vocab(text)
        self.set_input_shape(((self.tokenizer.length(), -1),))

    def tokenize(self: Embedding, text: str) -> Tokens:
        return self.tokenizer.tokenize(text)

    def get_one_hot(self: Embedding, entry: Tokens) -> Tensor:
        if len(entry) == 0:
            return np.array([[]])
        one_hot = np.zeros((self.tokenizer.length(), len(entry)))
        for i, token in enumerate(entry):
            one_hot[token, i] = 1
        return one_hot

    def get_embedded(self: Embedding, entry: Tokens) -> Tensor:
        if len(entry) == 0:
            return np.array([[]])
        embedded = np.empty((self.dim, len(entry)))
        for i in range(len(entry)):
            embedded[:, i] = self.W[:, entry[i]]
        return embedded

    def get_tokens(self: Embedding, entry: Tensor) -> Tokens:
        _, p = entry.shape
        tokens: Tokens = []
        for i in range(p):
            one_hot = entry[:, i]
            token = int(np.argmax(one_hot))
            tokens.append(token)
        return tokens

    def untokenize(self: Embedding, tokens: Tokens) -> str:
        return self.tokenizer.untokenize(tokens)

    def get_data(self: Embedding) -> SaveData:
        data = super().get_data()
        data["dim"] = self.dim
        data["W_prime"] = self.W_prime.flatten().tolist()
        tokenizer_data = self.tokenizer.get_data()
        tokenizer_data["class"] = self.tokenizer.__class__.__name__
        data["tokenizer"] = tokenizer_data
        return data

    def load_from_data(self: Embedding, data: SaveData) -> None:
        super().load_from_data(data)
        self.dim = data["dim"]
        self.W_prime = np.array(data["W_prime"]).reshape((self.input_shape[0][0], self.dim))
        tokenizer_class = data["tokenizer"]["class"]
        if tokenizer_class == "ByteTokenizer":
            self.tokenizer = ByteTokenizer()
            self.tokenizer.load_from_data(data["tokenizer"])
        elif tokenizer_class == "WordTokenizer":
            self.tokenizer = WordTokenizer()
            self.tokenizer.load_from_data(data["tokenizer"])
        else:
            raise MemoryError
