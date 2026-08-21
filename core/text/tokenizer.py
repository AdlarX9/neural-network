from __future__ import annotations
import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from ..utils.typing import Tokens, Tensor, SaveData


class Tokenizer(ABC):
    def __init__(self: Tokenizer) -> None:
        self.V: dict[Any, Any] = {}

    @abstractmethod
    def build_vocab(self: Tokenizer, corpus: str) -> None:
        pass

    @abstractmethod
    def tokenize(self: Tokenizer, text: str) -> Tokens:
        pass

    @abstractmethod
    def untokenize(self: Tokenizer, tokens: Tokens) -> str:
        pass

    def length(self: Tokenizer) -> int:
        return len(self.V)

    def get_one_hot(self: Tokenizer, token: int) -> Tensor:
        one_hot = np.zeros((len(self.V), 1))
        one_hot[token, 0] = 1
        return one_hot

    @abstractmethod
    def get_data(self: Tokenizer) -> SaveData:
        pass

    @abstractmethod
    def load_from_data(self: Tokenizer, data: SaveData) -> None:
        pass
