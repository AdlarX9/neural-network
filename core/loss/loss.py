from __future__ import annotations
from abc import ABC, abstractmethod
from ..utils.typing import TensorFlow


class Loss(ABC):
    def __init__(self: Loss) -> None:
        return

    @abstractmethod
    def get_loss(self: Loss, prediction: TensorFlow, answer: TensorFlow) -> tuple[float, ...]:
        pass

    @abstractmethod
    def get_gradient(self: Loss, prediction: TensorFlow, answer: TensorFlow) -> TensorFlow:
        pass
