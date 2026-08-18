from __future__ import annotations
from ..layer.layer import Layer
from typing import Callable
import numpy as np
from numpy.typing import NDArray


class Block(Layer):
    def __init__(self: Block, layers: list[Layer] = [], receive: tuple = (0,)) -> None:
        super().__init__(receive)
        self.layers: list[Layer] = layers

    def set_lr(self: Block, lr: float) -> None:
        super().set_lr(lr)
        for layer in self.layers:
            layer.set_lr(lr)

    def _distribute(self: Block, quantity: tuple, function: Callable) -> tuple:
        volume = list(quantity)
        for layer in self.layers:
            entry = tuple([volume[idx] for idx in layer.receive])
            output = list(function(layer, entry))
            for i in sorted(layer.receive, reverse=True):
                del volume[i]
            separator = min(layer.receive)
            volume[separator:separator] = output
        return tuple(volume)

    def set_input_shape(self: Block, input_shape: tuple) -> tuple:
        super().set_input_shape(input_shape)
        self.output_shape = self._distribute(input_shape, lambda layer, entry: layer.set_input_shape(entry))
        return self.output_shape

    def __call__(self: Block, entry: tuple, memorize: bool) -> tuple:
        super().__call__(entry, memorize)
        output = self._distribute(entry, lambda layer, entry: layer(entry, memorize))
        return output

    def backprop(self: Block, gradient: tuple) -> tuple:
        super().backprop(gradient)
        volume = list(gradient)
        for layer in reversed(self.layers):
            beginning, end = min(layer.receive), min(layer.receive) + len(layer.output_shape)
            input_slice = tuple(volume[beginning:end])
            del volume[beginning:end]
            output = layer.backprop(input_slice)
            losses: list[tuple[int, NDArray[np.float64]]] = [
                (el, output[idx]) for idx, el in enumerate(layer.receive)
            ]
            losses = sorted(losses, key=lambda x: x[0])
            for pos, loss in losses:
                volume.insert(pos, loss)
        return tuple(volume)

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
