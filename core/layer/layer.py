from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def check_shapes(shape1: tuple, shape2: tuple) -> bool:
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != -1 and shape2[i] != -1 and shape1[i] != shape2[i]:
            return False
    return True


class Layer:
    def __init__(self: Layer) -> None:
        self.lr: float = 0.0
        self.input_shape: tuple = ()
        self.output_shape: tuple = ()
        self.input: NDArray[np.float64] | None = None

    def set_lr(self: Layer, lr: float) -> None:
        self.lr = lr

    def set_input_shape(self: Layer, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        self.output_shape = self.input_shape
        return self.output_shape

    def get_dimensions(self: Layer) -> tuple[tuple, tuple]:
        return self.input_shape, self.output_shape

    def feed_forward(self: Layer, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry

    def compute(self: Layer, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        if not check_shapes(entry.shape, self.input_shape):
            print(entry.shape, self.input_shape)
            raise ValueError
        if memorize:
            self.input = entry
        return self.feed_forward(entry)

    def descend_gradient(self: Layer, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return gradient

    def backprop(self: Layer, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input_shape is None:
            raise MemoryError
        return self.descend_gradient(gradient)

    def get_data(self: Layer) -> dict:
        data = {
            "lr": self.lr,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }
        return data

    def load_from_data(self: Layer, data: dict) -> None:
        self.lr = data["lr"]
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]

    def save(self: Layer, name: str) -> None:
        from data import SaveHandler

        SaveHandler().save(self, name)

    def load(self: Layer, name: str) -> bool:
        from data import SaveHandler

        handler = SaveHandler()
        if not handler.has(name):
            return False
        layer = handler.load(name)
        if not isinstance(layer, self.__class__):
            raise TypeError(f"Expected {self.__class__.__name__}, " f"got {layer.__class__.__name__}")
        self.__dict__.clear()
        self.__dict__.update(layer.__dict__)
        return True
