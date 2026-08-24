from __future__ import annotations
from ..utils.typing import TrainData, Tokens, TensorFlow
from ..text.text_network import TextNetwork
from ..network.ddpm.ddpm import DDPM
import random
import numpy as np
import math


class Data:
    def __init__(self: Data, training_set: TrainData = [], test_set: TrainData = []) -> None:
        self.training_set: TrainData = training_set
        self.test_set: TrainData = test_set

    def get_samples(
        self: Data,
        text_network: TextNetwork,
        text: str,
        context_length: int = 256,
        stride: int = 128,
    ) -> list[tuple[Tokens, Tokens]]:
        tokens = text_network.tokenize(text)
        data: list[tuple[Tokens, Tokens]] = []
        if len(tokens) <= context_length:
            data.append((tokens[:-1], tokens[1:]))
        else:
            for i in range(0, len(tokens) - context_length, stride):
                data.append((tokens[i : i + context_length], tokens[i + 1 : i + context_length + 1]))
        random.shuffle(data)
        return data

    def build_tokens_data(self: Data, text_network: TextNetwork, data: list[tuple[Tokens, Tokens]]) -> None:
        new_data: TrainData = []
        show = bool(len(data) >= 1000)
        for i in range(len(data)):
            if show:
                progress = i / len(data) * 100
                print("One Hot progress: " f"{progress:.2f}%", end="\r")
            entry = text_network.get_one_hot(data[i][0])
            answer = text_network.get_one_hot(data[i][1])
            new_data.append(((entry,), (answer,)))
        if show:
            print("One Hot progress: 100.00%")
        del data
        self.training_set = new_data

    def build_transformer_data(
        self: Data,
        text_network: TextNetwork,
        text: str,
        context_length: int = 256,
        stride: int = 128,
    ) -> None:
        self.build_tokens_data(text_network, self.get_samples(text_network, text, context_length, stride))

    def build_ddpm_data(self: Data, ddpm: DDPM, data: list[tuple[TensorFlow, TensorFlow]]) -> None:
        self.training_set = []
        for text_one_hot, image in data:
            t = random.randint(1, ddpm.T)
            noise = np.random.randn(*image[0].shape)
            blurry_image = (
                math.sqrt(ddpm.get_alpha_bar(t)) * image[0] + math.sqrt(1 - ddpm.get_alpha_bar(t)) * noise
            )
            self.training_set.append(((blurry_image, text_one_hot[0], np.array([[t]])), (noise,)))
