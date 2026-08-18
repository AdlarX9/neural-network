from __future__ import annotations
from ..layer.layer import Layer
from typing import Callable


class Block(Layer):
    def __init__(self: Block, layers: list[Layer] = [], receive: tuple = (0,)) -> None:
        super().__init__(receive)
        self.layers: list[Layer] = layers

    def set_lr(self: Block, lr: float) -> None:
        super().set_lr(lr)
        for layer in self.layers:
            layer.set_lr(lr)

    def _distribute(self: Block, quantity: tuple, function: Callable) -> tuple:
        volume = quantity
        for layer in self.layers:
            entry = tuple([volume[idx] for idx in layer.receive])
            output = tuple(function(layer, entry))
            for i in reversed(layer.receive):
                if i + 1 == len(volume):
                    volume = volume[:i]
                else:
                    volume = volume[:i] + volume[i + 1 :]
            separator = layer.receive[0]
            if separator + 1 == len(volume):
                volume += output
            else:
                volume = volume[:separator] + output + volume[separator:]
        return volume

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
        current_gradient = gradient
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer._receive == 1:
                # Compatibilité avec le paradigme historique : un seul flux d’entrée / sortie.
                current_gradient = layer.backprop(current_gradient)
                continue

            # Nouveau paradigme multimodal : on reconstruit le tuple de sorties branchées
            # en conservant le comportement de la version legacy pour les réseaux classiques.
            input_slice = current_gradient[layer.receive[0] : layer.receive[0] + len(layer.output_shape)]
            output = layer.backprop(input_slice)
            rebuilt = list(current_gradient)
            for j in reversed(range(layer._receive)):
                pos = layer.receive[j]
                loss = output[j]
                if pos + 1 == len(rebuilt):
                    rebuilt = rebuilt[:pos] + [loss]
                else:
                    rebuilt = rebuilt[:pos] + [loss] + rebuilt[pos + 1 :]
            current_gradient = tuple(rebuilt)
        return current_gradient

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
