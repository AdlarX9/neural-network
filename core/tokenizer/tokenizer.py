from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    def __init__(self: Tokenizer) -> None:
        self.V: dict = {}

    @abstractmethod
    def build_vocab(self: Tokenizer, corpus: str) -> None:
        pass

    @abstractmethod
    def tokenize(self: Tokenizer, text: str) -> list[int]:
        pass

    @abstractmethod
    def untokenize(self: Tokenizer, tokens: list[int]) -> str:
        pass

    def length(self: Tokenizer) -> int:
        return len(self.V)

    def get_one_hot(self: Tokenizer, token: int) -> NDArray[np.float64]:
        one_hot = np.zeros((len(self.V), 1))
        one_hot[token, 0] = 1
        return one_hot

    @abstractmethod
    def get_data(self: Tokenizer) -> dict:
        pass

    @abstractmethod
    def load_from_data(self: Tokenizer, data: dict) -> None:
        pass
