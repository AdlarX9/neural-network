from __future__ import annotations
from .tokenizer import Tokenizer
import re
import unicodedata


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("'", " ")
    text = text.replace("’", " ")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WordTokenizer(Tokenizer):
    def __init__(self: WordTokenizer) -> None:
        self.V: dict[str, int] = {}

    def build_vocab(self: WordTokenizer, corpus: str) -> None:
        words = normalize(corpus).split(" ")
        index = len(self.V)
        for word in words:
            if word not in self.V:
                self.V[word] = index
                index += 1

    def tokenize(self: WordTokenizer, text: str) -> list[int]:
        words = normalize(text).split(" ")
        return [self.V[word] for word in words]

    def untokenize(self: WordTokenizer, tokens: list[int]) -> str:
        words: list[str] = []
        for token in tokens:
            word = next(k for k, v in self.V.items() if v == token)
            words.append(word)
        return " ".join(words)

    def get_data(self: WordTokenizer) -> dict:
        data = {"V": self.V}
        return data

    def load_from_data(self: WordTokenizer, data: dict) -> None:
        self.V = data["V"]
