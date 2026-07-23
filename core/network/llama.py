from __future__ import annotations
from ..exit.proba_exit import ProbaExit
from ..layer.embedding import Embedding
from ..transform.rms_norm import RMSNorm
from ..layer.layer import Layer
from ..block.one_hot_maker import OneHotMaker
from ..block.res import Res
from ..block.block import Block
from ..layer.mha import MHA
from ..block.linear import Linear
from ..block.swiglu import SwiGLU
from .word_network import WordNetwork


class LLaMA(WordNetwork):
    def __init__(
        self: LLaMA,
        head_numbers: list[int] = [],
        embedding: Embedding | None = None,
        lr: float = 0.0001,
    ) -> None:
        if embedding is None:
            return
        layers: list[Layer] = [embedding]
        for H in head_numbers:
            layers.append(Res(Block(RMSNorm(), MHA(H))))
            layers.append(Res(Block(RMSNorm(), SwiGLU(int(8 / 3 * embedding.dim)), Linear(embedding.dim))))
        layers += [RMSNorm(), OneHotMaker(embedding)]
        super().__init__(layers, (embedding.tokenizer.length(), -1), lr, ProbaExit(axis=0))

    def predict_next_word(self: LLaMA, beginning: list[str]) -> str:
        predictions = self.compute_words(beginning)
        return predictions[-1]
