from __future__ import annotations
from core.exit.exit_loss import ExitLoss
from .network import Network
from ..block.recurrent import Recurrent
from ..block.decoder import Decoder
from ..layer.embedding import Embedding
from ..layer.lstm import LSTM
from ..utils.tokenizer import Tokenizer
import numpy as np


class LSTMNetwork(Network):
    def __init__(
        self: LSTMNetwork,
        embedding: Embedding | None = None,
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        layers = []
        input_shape = (0,)
        if embedding is not None:
            layers = [embedding, Recurrent(LSTM()), Decoder(embedding)]
            input_shape = (embedding.input_shape[0], -1)
        super().__init__(layers=layers, input_shape=input_shape, lr=lr, exit_loss=exit_loss)

    def predict_next_word(self: LSTMNetwork, beginning: list[str]) -> str:
        tokenizer = Tokenizer()
        one_hot_sentence = None
        for word in beginning:
            if one_hot_sentence is None:
                one_hot_sentence = tokenizer.get_one_hot(word).reshape(-1, 1)
            else:
                one_hot_sentence = np.hstack((one_hot_sentence, tokenizer.get_one_hot(word).reshape(-1, 1)))
        if one_hot_sentence is None:
            raise ValueError
        for layer in self.layers:
            if isinstance(layer, Recurrent):
                layer.reset_data()
        one_hot_prediction = self.compute(one_hot_sentence)
        prediction = tokenizer.get_word(int(np.argmax(one_hot_prediction)))
        return prediction
