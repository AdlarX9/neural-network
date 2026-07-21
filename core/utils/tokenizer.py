from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import re
import unicodedata

accepted_chars = "abcdefghijklmnopqrstuvwxyz "


def normalize(text: str) -> str:
    text = text.lower()

    # apostrophes -> espaces
    text = text.replace("'", " ")
    text = text.replace("’", " ")

    # enlève les accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # garde uniquement a-z et espaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # espaces multiples
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class Tokenizer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self: Tokenizer) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.V: dict[str, int] = {}

    def length(self: Tokenizer) -> int:
        return len(self.V)

    def split_text(self: Tokenizer, text: str):
        text = normalize(text)
        split = text.split(" ")
        return split

    def build_vocab(self: Tokenizer, corpus: str) -> None:
        split = self.split_text(corpus)
        index = len(self.V)
        for word in split:
            if word not in self.V:
                self.V[word] = index
                index += 1
    
    def get_word(self: Tokenizer, index: int) -> str:
        return next(k for k, v in self.V.items() if v == index)

    def get_index(self: Tokenizer, word: str) -> int:
        if word not in self.V:
            raise ValueError("Word not in vocabulary:", word)
        return self.V[word]

    def get_one_hot(self: Tokenizer, word: str) -> NDArray[np.float64]:
        index = self.get_index(word)
        one_hot = np.zeros((len(self.V), 1))
        one_hot[index, 0] = 1
        return one_hot
