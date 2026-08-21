from __future__ import annotations
from ..utils.typing import Shape, ShapeFlow, Tensor, TensorFlow, Receive, SaveData, ParamGrad
from typing import Any


def check_shape(shape1: Shape, shape2: Shape) -> bool:
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != -1 and shape2[i] != -1 and shape1[i] != shape2[i]:
            return False
    return True


def check_shapes(shape1: ShapeFlow, shape2: ShapeFlow) -> bool:
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if not check_shape(shape1[i], shape2[i]):
            return False
    return True


class Layer:
    def __init__(self: Layer, receive: Receive = (0,)) -> None:
        self.lr: float = 0.0
        self.input_shape: ShapeFlow = ((),)
        self.output_shape: ShapeFlow = ((),)
        self.input: TensorFlow | None = None
        if not hasattr(self, "_receive"):
            self._receive: int = 1
        if self._receive != -1 and len(receive) != self._receive:
            raise ValueError(self._receive, receive)
        self.receive: Receive = receive
        self._id: int = -1
        self.parameters: list[str] = []

    def set_input_shape(self: Layer, input_shape: ShapeFlow) -> ShapeFlow:
        if self._receive != -1 and len(input_shape) != self._receive:
            raise ValueError(input_shape, self._receive, self.receive)
        self.input_shape = input_shape
        self.output_shape = self.input_shape
        return self.output_shape

    def feed_forward(self: Layer, entry):
        return entry

    def __call__(self: Layer, entry: TensorFlow, memorize: bool = False) -> TensorFlow:
        entry_shape: ShapeFlow = tuple(el.shape for el in entry)
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
        if isinstance(output, tuple):
            return output
        else:
            return (output,)

    def descend_gradient(self: Layer, gradient):
        return gradient

    def params_gradient(self: Layer, gradient) -> ParamGrad:
        return {}

    def backprop(self: Layer, gradient: TensorFlow) -> tuple[TensorFlow, ParamGrad]:
        if self.input_shape is None:
            raise MemoryError
        output: TensorFlow | Tensor = ()
        params_input = gradient
        if len(self.output_shape) == 1:
            params_input = gradient[0]
            output = self.descend_gradient(gradient[0])
        else:
            output = self.descend_gradient(gradient)
        if isinstance(output, tuple):
            new_gradient: TensorFlow = output
        else:
            new_gradient: TensorFlow = (output,)
        gradients = self.params_gradient(params_input)
        prefix = str(self._id) + "."
        gradients = {prefix + key: value for key, value in gradients.items()}
        return new_gradient, gradients

    def get_data(self: Layer) -> SaveData:
        data = {
            "lr": self.lr,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "receive": self.receive,
            "_receive": self._receive,
            "_id": self._id,
            "parameters": self.parameters,
        }
        return data

    def load_from_data(self: Layer, data: SaveData) -> None:
        self.lr = data["lr"]
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]
        self.receive = data["receive"]
        self._receive = data["_receive"]
        self._id = data["_id"]
        self.parameters = data["parameters"]

    def set_lr(self: Layer, lr: float) -> None:
        self.lr = lr

    def set_id(self: Layer, id: int = 0) -> int:
        if self._id != -1:
            return 0
        self._id = id
        return id + 1

    def get_dimensions(self: Layer) -> tuple[ShapeFlow, ShapeFlow]:
        return self.input_shape, self.output_shape

    def get_params_data(self: Layer) -> list[dict[str, Any]]:
        data = [
            {
                "id": self._id,
                "parameters": self.parameters,
            }
        ]
        for param_name in self.parameters:
            data[0][param_name] = getattr(self, param_name)
        return data

    def get_layer_by_id(self: Layer, id: int) -> Layer | None:
        if self._id == id:
            return self
        return None

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
