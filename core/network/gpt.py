from __future__ import annotations

from core.exit.exit_loss import ExitLoss
from core.layer.layer import Layer
from .text_network import TextNetwork
from ..layer.embedding import Embedding


class GPT(TextNetwork):
    def __init__(
        self: GPT,
        layers: list[Layer] = [],
        input_shape: tuple = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
        embedding: Embedding = Embedding(),
    ) -> None:
        super().__init__(layers, input_shape, lr, exit_loss, embedding)

    def predict_next_token(self: GPT, beginning: str) -> str:
        one_hot = self.get_embedded(self.tokenize(beginning))
        result = self((one_hot,))
        prediction = self.untokenize(self.get_tokens(result[0])[-1:])
        return prediction
