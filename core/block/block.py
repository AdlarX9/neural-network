from __future__ import annotations
from ..layer.layer import Layer
import numpy as np
from numpy.typing import NDArray


class Block(Layer):
    def __init__(self: Block, *layers: Layer):
        super().__init__()
        self.layers: list[Layer] = list(layers)

    def set_lr(self: Block, lr: float) -> None:
        super().set_lr(lr)
        for layer in self.layers:
            layer.set_lr(lr)

    def set_input_shape(self: Block, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        volume = input_shape
        for layer in self.layers:
            volume = layer.set_input_shape(volume)
        self.output_shape = volume
        return self.output_shape

    def compute(self: Block, entry: NDArray[np.float64], memorize: bool) -> NDArray[np.float64]:
        super().compute(entry, memorize)
        for layer in self.layers:
            entry = layer.compute(entry, memorize)
        return entry

    def backprop(self: Block, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        super().backprop(gradient)
        for i in reversed(range(len(self.layers))):
            gradient = self.layers[i].backprop(gradient)
        return gradient

    def get_data(self: Block) -> tuple[list[int], list[float], list[str]]:
        int_list = [len(self.input_shape), len(self.output_shape)] + list(
            self.input_shape + self.output_shape
        )
        float_list = [self.lr]
        string_list = []
        int_list.append(len(self.layers))
        for layer in self.layers:
            sub_int, sub_float, sub_string = layer.get_data()
            string_list += [layer.__class__.__name__]
            int_list += [len(sub_int), len(sub_float), len(sub_string)] + sub_int
            float_list += sub_float
            string_list += sub_string
        return int_list, float_list, string_list

    def load_from_data(
        self: Block,
        int_list: list[int],
        float_list: list[float],
        string_list: list[str],
        layer_types: dict[str, type[Layer]] = {},
    ) -> None:
        len_input_shape = int_list.pop(0)
        len_output_shape = int_list.pop(0)
        self.input_shape = tuple(int_list[:len_input_shape])
        del int_list[:len_input_shape]
        self.output_shape = tuple(int_list[:len_output_shape])
        del int_list[:len_output_shape]
        self.lr = float_list.pop(0)
        len_layers = int_list.pop(0)
        self.layers = []
        for _ in range(len_layers):
            class_name = string_list.pop(0)
            len_sub_int = int_list.pop(0)
            len_sub_float = int_list.pop(0)
            len_sub_string = int_list.pop(0)
            new_layer = layer_types[class_name]()
            if isinstance(new_layer, Block):
                new_layer.load_from_data(
                    int_list[:len_sub_int], float_list[:len_sub_float], string_list[:len_sub_string], layer_types
                )
            else:
                new_layer.load_from_data(
                    int_list[:len_sub_int], float_list[:len_sub_float], string_list[:len_sub_string]
                )
            del int_list[:len_sub_int]
            del float_list[:len_sub_float]
            del string_list[:len_sub_string]
            self.layers.append(new_layer)
