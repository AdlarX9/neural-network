from __future__ import annotations
from .tokenizer import Tokenizer


class ByteTokenizer(Tokenizer):
    def __init__(self: ByteTokenizer, vocab_size: int = 32_768) -> None:
        self.V: dict[tuple[int, ...], int] = {}
        self.R: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.vocab_size = vocab_size

    def build_vocab(self: ByteTokenizer, corpus: str) -> None:
        show = False
        if len(corpus) >= 10_000:
            show = True
        byte_values = list(corpus.encode("utf-8"))
        self.V = {(i,): i for i in range(256)}
        self.R = []

        tokens = [(b,) for b in byte_values]
        next_id = 256
        while next_id < self.vocab_size:
            if show:
                print(
                    "Vocabulary progress:", round(next_id / min(self.vocab_size, len(byte_values)) * 100, 2), "%", end="\r"
                )

            # Compte les paires adjacentes
            pair_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda pair: pair_counts[pair])
            left, right = best_pair
            merged = left + right

            # Ajout au vocabulaire et aux règles
            self.V[merged] = next_id
            self.R.append(best_pair)

            # Application de la fusion
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            next_id += 1
        if show:
            print("Vocabulary progress: 100.00 %")

    def tokenize(self: ByteTokenizer, text: str) -> list[int]:
        show = False
        if len(text) >= 10_000:
            show = True
        tokens = [(b,) for b in text.encode("utf-8")]
        for idx, (left, right) in enumerate(self.R):
            if show:
                print("Tokenizing progress:", round(idx / len(self.R) * 100, 2), "%", end="\r")
            merged = left + right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        if show:
            print("Tokenizing progress: 100.00 %")
        return [self.V[token] for token in tokens]

    def untokenize(self: ByteTokenizer, tokens: list[int]) -> str:
        id_to_token = {v: k for k, v in self.V.items()}
        byte_values = []
        for token in tokens:
            byte_values += id_to_token[token]
        return bytes(byte_values).decode("utf-8")

    def get_data(self: ByteTokenizer) -> tuple[list[int], list[float], list[str]]:
        int_list: list[int] = []
        int_list.append(self.vocab_size)
        int_list.append(len(self.R))
        for left, right in self.R:
            int_list.append(len(left))
            int_list.extend(left)
            int_list.append(len(right))
            int_list.extend(right)
        int_list.append(len(self.V))
        for key, value in self.V.items():
            int_list.append(value)
            int_list.append(len(key))
            int_list.extend(key)

        return int_list, [], []

    def load_from_data(
        self: ByteTokenizer,
        int_list: list[int],
        float_list: list[float],
        string_list: list[str],
    ) -> None:
        self.V = {}
        self.R = []
        idx = 0
        self.vocab_size = int_list[idx]
        idx += 1
        r_count = int_list[idx]
        idx += 1
        for _ in range(r_count):
            left_len = int_list[idx]
            idx += 1
            left = tuple(int_list[idx : idx + left_len])
            idx += left_len
            right_len = int_list[idx]
            idx += 1
            right = tuple(int_list[idx : idx + right_len])
            idx += right_len
            self.R.append((left, right))
        v_count = int_list[idx]
        idx += 1
        for _ in range(v_count):
            value = int_list[idx]
            idx += 1
            key_len = int_list[idx]
            idx += 1
            key = tuple(int_list[idx : idx + key_len])
            idx += key_len
            self.V[key] = value
