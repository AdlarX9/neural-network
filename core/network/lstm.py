from __future__ import annotations
from core.exit.exit_loss import ExitLoss
from .word_network import WordNetwork
from ..block.recurrent import Recurrent
from ..block.one_hot_maker import OneHotMaker
from ..layer.embedding import Embedding
from ..layer.lstm import LSTM


class LSTMNetwork(WordNetwork):
    def __init__(
        self: LSTMNetwork,
        embedding: Embedding | None = None,
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        layers = []
        input_shape = (0,)
        if embedding is not None:
            layers = [embedding, Recurrent(LSTM()), OneHotMaker(embedding)]
            input_shape = (embedding.input_shape[0], -1)
        super().__init__(layers=layers, input_shape=input_shape, lr=lr, exit_loss=exit_loss)

    def predict_next_word(self: LSTMNetwork, beginning: list[str]) -> str:
        for layer in self.layers:
            if isinstance(layer, Recurrent):
                layer.reset_data()
        one_hot_beginning = self.get_one_hot(beginning)
        one_hot_prediction = self.compute(one_hot_beginning)
        prediction = self.get_words(one_hot_prediction)
        return prediction[0]
