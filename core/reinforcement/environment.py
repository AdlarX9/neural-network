from __future__ import annotations
from abc import ABC, abstractmethod
from ..utils.typing import TensorFlow


class Environment(ABC):
    def __init__(self: Environment) -> None:
        self.state: TensorFlow = ()

    @abstractmethod
    def update(self: Environment, action: TensorFlow) -> None:
        pass
