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

    def get_data(self: Block) -> dict:
        data = super().get_data()
        layers_data = []
        for layer in self.layers:
            layer_data = layer.get_data()
            layer_data["class"] = layer.__class__.__name__
            layers_data.append(layer_data)
        data["layers"] = layers_data
        return data

    def load_from_data(self: Block, data: dict, layer_types: dict[str, type[Layer]] = {}) -> None:
        super().load_from_data(data)
        self.layers = []
        for layer_data in data["layers"]:
            class_name = layer_data["class"]
            new_layer = layer_types[class_name]()
            if isinstance(new_layer, Block):
                new_layer.load_from_data(layer_data, layer_types)
            else:
                new_layer.load_from_data(layer_data)
            self.layers.append(new_layer)
