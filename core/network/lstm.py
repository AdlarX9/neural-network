from __future__ import annotations
from ..text.text_network import TextNetwork
from ..block.recurrent import Recurrent
from ..block.one_hot_maker import OneHotMaker
from ..parameterized.embedding import Embedding
from ..parameterized.lstm import LSTM
from ..basics.layer import Layer


class LSTMNetwork(TextNetwork):
    def __init__(
        self: LSTMNetwork,
        embedding: Embedding | None = None,
        lr: float = 0.001,
        more_layers: list[Layer] = [],
    ) -> None:
        if embedding is None:
            return
        layers = []
        input_shape = (0,)
        if embedding is not None:
            layers = [Recurrent([LSTM()]), OneHotMaker(embedding)] + more_layers
            input_shape = (embedding.tokenizer.length(), -1)
        super().__init__(layers=layers, embedding=embedding)
        self.set_lr(lr)
        self.set_input_shape((input_shape,))

    def predict_next_token(self: LSTMNetwork, beginning: str) -> str:
        for layer in self.layers:
            if isinstance(layer, Recurrent):
                layer.reset_data()
        one_hot_beginning = self.get_one_hot(self.tokenize(beginning))
        one_hot_prediction = self((one_hot_beginning,))
        prediction = self.untokenize(self.get_tokens(one_hot_prediction[0]))
        return prediction
