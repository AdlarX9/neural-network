from __future__ import annotations

from core.exit.exit_loss import ExitLoss
from core.layer.layer import Layer
from .word_network import WordNetwork


class GPT(WordNetwork):
    def __init__(
        self: GPT,
        layers: list[Layer] = [],
        input_shape: tuple = (0,),
        lr: float = 0.0001,
        exit_loss: ExitLoss = ExitLoss(),
    ) -> None:
        super().__init__(layers, input_shape, lr, exit_loss)

    def predict_next_word(self: GPT, beginning: list[str]) -> str:
        predictions = self.compute_words(beginning)
        return predictions[-1]
