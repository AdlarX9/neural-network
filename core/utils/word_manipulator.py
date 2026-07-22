from __future__ import annotations
from .tokenizer import Tokenizer
import numpy as np
from numpy.typing import NDArray


class WordManipulator:
    def __init__(self: WordManipulator) -> None:
        self.tokenizer = Tokenizer()

    def get_one_hot(self: WordManipulator, entry: list[str]) -> NDArray[np.float64]:
        one_hot = None
        for token in entry:
            representation = self.tokenizer.get_one_hot(token).reshape(-1, 1)
            if one_hot is None:
                one_hot = representation
            else:
                one_hot = np.hstack((one_hot, representation))
        if one_hot is None:
            one_hot = np.array([[]])
        return one_hot

    def get_words(self: WordManipulator, entry: NDArray[np.float64]) -> list[str]:
        _, p = entry.shape
        words: list[str] = []
        for i in range(p):
            one_hot = entry[:, i]
            index = int(np.argmax(one_hot))
            word = self.tokenizer.get_word(index)
            words.append(word)
        return words
