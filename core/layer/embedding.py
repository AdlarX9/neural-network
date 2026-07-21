from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from ..utils.tokenizer import Tokenizer
from ..utils.functions import softmax
from graphics import ConsoleVisualization
import math


class Embedding(Layer):
    def __init__(self: Embedding, dim: int = 100) -> None:
        super().__init__()
        self.dim = dim
        self.W = np.array([[]])
        self.W_prime = np.array([[]])
        self.tokenizer = Tokenizer()

    def set_input_shape(self: Embedding, input_shape: tuple[int, int]) -> tuple[int, int]:
        if len(input_shape) != 2 and input_shape[0] != self.tokenizer.length():
            raise ValueError(
                "Expected dimension does not fit Tokenizer requiremenents:",
                (self.tokenizer.length(), 1),
                "!=",
                input_shape,
            )
        self.input_shape = input_shape
        self.W = np.random.normal(
            -1 / np.sqrt(self.dim),
            1 / np.sqrt(self.dim),
            (self.dim, Tokenizer().length()),
        )
        self.W_prime = np.random.normal(
            -1 / np.sqrt(self.dim),
            1 / np.sqrt(self.dim),
            (Tokenizer().length(), self.dim),
        )
        self.output_shape = (self.dim, self.input_shape[1])
        return self.output_shape

    def feed_forward(self: Embedding, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.W @ entry

    def descend_gradient(self: Embedding, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = self.W.T @ gradient
        self.W -= self.lr * gradient @ self.input.T
        return new_gradient

    def cbow_training(self: Embedding, text: str, window: int = 7, batch: int = 10) -> None:
        words = self.tokenizer.split_text(text)
        dashboard = ConsoleVisualization(batch, len(words) - 2 * window)
        max_length = 500
        corrects = []
        losses = []
        for batch_index in range(1, batch + 1):
            for i in range(window, len(words) - window):
                target = words[i]
                target_index = self.tokenizer.get_index(target)
                answer = self.tokenizer.get_one_hot(target)

                # Context vector
                contexts = words[i - window : i] + words[i + 1 : i + window + 1]
                one_hots = [self.tokenizer.get_one_hot(word) for word in contexts]
                embedded_contexts = [
                    self.W[:, self.tokenizer.get_index(context)].reshape(-1, 1) for context in contexts
                ]
                embedded_context = sum(embedded_contexts) / len(embedded_contexts)

                # Compute score
                score = self.W_prime @ embedded_context
                probability = softmax(score)

                # Learn
                gradient = probability - answer
                new_gradient = self.W_prime.T @ gradient
                self.W_prime -= self.lr * gradient @ embedded_context.T  # type: ignore
                self.W -= self.lr * new_gradient @ sum(one_hots).T / (2 * window)  # type: ignore

                # Dashboard
                correct = bool(np.argmax(probability) == np.argmax(answer))
                corrects.append(correct)
                if len(corrects) > max_length:
                    corrects.pop(0)
                loss = -math.log(probability[target_index, 0])
                losses.append(loss)
                if len(losses) > max_length:
                    losses.pop(0)
                dashboard.update(
                    batch_index,
                    i + 1 - window,
                    sum(losses) / len(losses),
                    corrects.count(True) / len(corrects),
                )

    def custom_training(self: Embedding, text: str, window: int = 3, batch: int = 10):
        K = 1e-8
        words = self.tokenizer.split_text(text)
        word_counts = np.zeros((self.tokenizer.length()))
        for word in words:
            word_counts[self.tokenizer.get_index(word)] += 1
        word_freq = word_counts / len(words)
        for batch_index in range(1, batch + 1):
            for i in range(window, len(words) - window):
                center = words[i]
                center_idx = self.tokenizer.get_index(center)
                center_freq = word_freq[center_idx]
                embedded_center = self.W[:, center_idx]
                all_indices = set([i for i in range(self.tokenizer.length())])
                all_indices.remove(center_idx)
                for j in range(-window, window + 1):
                    if j == 0:
                        continue
                    context_word = words[i - j]
                    context_idx = self.tokenizer.get_index(context_word)
                    all_indices.remove(context_idx)
                    embedded_context = self.W[:, context_idx]
                    cosine = self.cosine_similarity(embedded_context, embedded_center)
                    force = K / (center_freq * word_freq[context_idx] * j ** 2 + 1e-12) * (1 - cosine)
                    correction = (embedded_center - embedded_context)
                    correction /= np.linalg.norm(correction)
                    correction *= force
                    self.W[:, context_idx] += self.lr * correction
                opposite_force = embedded_center
                opposite_force /= np.linalg.norm(opposite_force)
                opposite_force *= 2 * K / (len(all_indices) * center_freq * window + 1e-12)
                for index in all_indices:
                    self.W[:, index] -= self.lr * opposite_force

    def cosine_similarity(self: Embedding, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def distance(self: Embedding, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.sum((b - a) ** 2))

    def estimate(self: Embedding, formula: str) -> str:
        split = formula.split(" ")
        if len(split) % 2 == 0:
            raise ValueError("Invalid Formula")
        words = split[::2]
        symbols = split[1::2]
        vectors = [self.W[:, self.tokenizer.get_index(word)].copy() for word in words]
        output = vectors[0].copy()
        index = 1
        for symbol in symbols:
            if symbol == "+":
                output += vectors[index]
            elif symbol == "-":
                output -= vectors[index]
            index += 1
        best_word = ""
        best_similarity = -math.inf
        for word in self.tokenizer.V:
            embedded_word = self.W[:, self.tokenizer.get_index(word)]
            similarity = self.cosine_similarity(output, embedded_word)
            if similarity > best_similarity:
                best_similarity = similarity
                best_word = word
        return best_word

    def look_around(self: Embedding, word: str, number: int = 15) -> None:
        embedded_word = self.W[:, self.tokenizer.get_index(word)]
        words = [
            (
                random_word,
                self.cosine_similarity(embedded_word, self.W[:, self.tokenizer.get_index(random_word)]),
            )
            for random_word in self.tokenizer.V.keys()
        ]
        words = sorted(words, key=lambda x: x[1], reverse=True)

        print("")
        print("Closest to", word, ":")
        for i in range(number):
            first = str(i + 1) + "."
            space1 = " " * (5 - len(first))
            second = words[i][0]
            space2 = " " * (20 - len(second))
            third = " | "
            fourth = str(round(words[i][1], 4))
            print(first + space1 + second + space2 + third + fourth)
        print("")

    def predict(self: Embedding, words: list[str], number: int = 10) -> None:
        context = (
            self.W[:, [self.tokenizer.get_index(word) for word in words]]
            .mean(axis=1, keepdims=True)
            .reshape(-1, 1)
        )
        prediction = self.W_prime @ context
        probability = softmax(prediction)
        predictions = [
            (self.tokenizer.get_word(i), float(probability[i, 0])) for i in range(self.tokenizer.length())
        ]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        print("")
        print("Closest to", *words, ":")
        for i in range(number):
            first = str(i + 1) + "."
            space1 = " " * (5 - len(first))
            second = predictions[i][0]
            space2 = " " * (20 - len(second))
            third = " | "
            fourth = str(round(predictions[i][1], 4))
            print(first + space1 + second + space2 + third + fourth)
        print("")

    def get_data(self: Embedding) -> tuple[list[int], list[float], list[str]]:
        int_list = (
            list(self.input_shape)
            + list(self.output_shape)
            + [self.dim, self.tokenizer.length()]
            + list(self.tokenizer.V.values())
        )
        float_list = [self.lr] + self.W.flatten().tolist() + self.W_prime.flatten().tolist()
        string_list = list(self.tokenizer.V.keys())
        return int_list, float_list, string_list

    def load_from_data(
        self: Embedding,
        int_list: list[int],
        float_list: list[float],
        string_list: list[str],
    ) -> None:
        self.input_shape = tuple(int_list[:2])
        del int_list[:2]
        self.output_shape = tuple(int_list[:2])
        del int_list[:2]
        self.dim = int_list.pop(0)
        tokenizer_length = int_list.pop(0)
        self.lr = float_list.pop(0)
        self.W = np.array(float_list[: tokenizer_length * self.dim]).reshape(self.dim, tokenizer_length)
        del float_list[: tokenizer_length * self.dim]
        self.W_prime = np.array(float_list).reshape(tokenizer_length, self.dim)
        if self.tokenizer.length() != tokenizer_length:
            self.tokenizer.V = {}
            for i in range(tokenizer_length):
                self.tokenizer.V[string_list[i]] = int_list[i]
