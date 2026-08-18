from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def check_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> bool:
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != -1 and shape2[i] != -1 and shape1[i] != shape2[i]:
            return False
    return True


def check_shapes(shape1: tuple[tuple[int, ...]], shape2: tuple[tuple[int, ...]]) -> bool:
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if not check_shape(shape1[i], shape2[i]):
            return False
    return True


class Layer:
    def __init__(self: Layer, receive: tuple[int, ...] = (0,)) -> None:
        self.lr: float = 0.0
        self.input_shape: tuple[tuple[int, ...]] = ((),)
        self.output_shape: tuple[tuple] = ((),)
        self.input: tuple[NDArray[np.float64], ...] | None = None
        self._receive: int = 1
        if len(receive) != self._receive:
            raise ValueError(self._receive, receive)
        self.receive: tuple[int, ...] = receive
        for i in range(1, self._receive):
            if self.receive[i - 1] >= self.receive[i]:
                raise ValueError

    def set_lr(self: Layer, lr: float) -> None:
        self.lr = lr

    def set_input_shape(self: Layer, input_shape: tuple) -> tuple:
        if len(input_shape) != self._receive:
            raise ValueError(input_shape, self._receive, self.receive)
        self.input_shape = input_shape
        self.output_shape = self.input_shape
        return self.output_shape

    def get_dimensions(self: Layer) -> tuple[tuple, tuple]:
        return self.input_shape, self.output_shape

    def feed_forward(self: Layer, entry):
        return entry

    def __call__(self: Layer, entry: tuple, memorize: bool) -> tuple:
        entry_shape: tuple[tuple[int, ...]] = tuple([el.shape for el in entry])  # type: ignore
        if not check_shapes(entry_shape, self.input_shape):
            print(entry_shape, self.input_shape)
            raise ValueError
        if memorize:
            self.input = entry
        output = None
        if self._receive == 1:
            output = self.feed_forward(entry[0])
        else:
            output = self.feed_forward(entry)
        if type(output) == tuple:
            return output
        else:
            return (output,)

    def descend_gradient(self: Layer, gradient):
        return gradient

    def backprop(self: Layer, gradient: tuple) -> tuple:
        if self.input_shape is None:
            raise MemoryError
        output = None
        if self._receive == 1:
            output = self.descend_gradient(gradient[0])
        else:
            output = self.descend_gradient(gradient)
        if type(output) == tuple:
            return output
        else:
            return (output,)

    def get_data(self: Layer) -> dict:
        data = {
            "lr": self.lr,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "receive": self.receive,
            "_receive": self._receive,
        }
        return data

    def load_from_data(self: Layer, data: dict) -> None:
        self.lr = data["lr"]
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]
        self.receive = data["receive"]
        self._receive = data["_receive"]

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
