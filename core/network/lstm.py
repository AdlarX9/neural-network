from __future__ import annotations
from core.exit.exit_loss import ExitLoss
from .text_network import TextNetwork
from ..block.recurrent import Recurrent
from ..block.one_hot_maker import OneHotMaker
from ..layer.embedding import Embedding
from ..layer.lstm import LSTM


class LSTMNetwork(TextNetwork):
    def __init__(
        self: LSTMNetwork,
        embedding: Embedding | None = None,
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        if embedding is None:
            return
        layers = []
        input_shape = (0,)
        if embedding is not None:
            layers = [Recurrent(LSTM()), OneHotMaker(embedding)]
            input_shape = (embedding.dim, -1)
        super().__init__(
            layers=layers, input_shape=input_shape, lr=lr, exit_loss=exit_loss, embedding=embedding
        )

    def predict_next_token(self: LSTMNetwork, beginning: str) -> str:
        for layer in self.layers:
            if isinstance(layer, Recurrent):
                layer.reset_data()
        one_hot_beginning = self.get_embedded(self.tokenize(beginning))
        one_hot_prediction = self.compute(one_hot_beginning)
        prediction = self.untokenize(self.get_tokens(one_hot_prediction))
        return prediction
