from __future__ import annotations
from ..exit.proba_exit import ProbaExit
from ..layer.embedding import Embedding
from ..block.transformer import TransformerBlock
from ..transform.rms_norm import RMSNorm
from ..layer.layer import Layer
from ..block.one_hot_maker import OneHotMaker
from .word_network import WordNetwork


class GPT(WordNetwork):
    def __init__(
        self: GPT,
        head_numbers: list[int] = [],
        embedding: Embedding | None = None,
        lr: float = 0.0001,
    ) -> None:
        if embedding is None:
            return
        layers: list[Layer] = [embedding]
        for H in head_numbers:
            layers.append(TransformerBlock(H))
        layers += [RMSNorm(), OneHotMaker(embedding)]
        super().__init__(layers, (embedding.tokenizer.length(), -1), lr, ProbaExit(axis=0))
    
    def predict_next_word(self: GPT, beginning: list[str]) -> str:
        predictions = self.compute_words(beginning)
        return predictions[-1]
