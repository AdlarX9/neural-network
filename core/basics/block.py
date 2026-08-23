from __future__ import annotations
from .layer import Layer
from typing import Any, Callable
from ..utils.typing import ShapeFlow, TensorFlow, Tensor, SaveData, Receive, ParamGrad


class Block(Layer):
    def __init__(
        self: Block,
        layers: list[Layer] = [],
        receive: Receive = (0,),
    ) -> None:
        super().__init__(receive)
        self.layers: list[Layer] = layers

    def set_lr(self: Block, lr: float) -> None:
        super().set_lr(lr)
        for layer in self.layers:
            layer.set_lr(lr)

    def set_id(self: Block, id: int = 0) -> int:
        if self._id != -1:
            return 0
        id = super().set_id(id)
        for layer in self.layers:
            id = layer.set_id(id)
        return id

    def _distribute(
        self: Block,
        quantity: tuple[Any, ...],
        function: Callable[[Layer, tuple[Any, ...]], tuple[Any, ...]],
    ) -> tuple[Any, ...]:
        volume = list(quantity)
        for layer in self.layers:
            entry = tuple([volume[idx] for idx in layer.receive])
            output = list(function(layer, entry))
            for i in sorted(layer.receive, reverse=True):
                del volume[i]
            separator = min(layer.receive)
            volume[separator:separator] = output
        return tuple(volume)

    def set_input_shape(self: Block, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        self.output_shape = self._distribute(input_shape, lambda layer, entry: layer.set_input_shape(entry))
        return self.output_shape

    def __call__(self: Block, entry: TensorFlow, memorize: bool = False) -> TensorFlow:
        super().__call__(entry, memorize)
        output = self._distribute(entry, lambda layer, entry: layer(entry, memorize))
        return output

    def backprop(self: Block, gradient: TensorFlow) -> tuple[TensorFlow, ParamGrad]:
        super().backprop(gradient)
        volume = list(gradient)
        params: ParamGrad = {}
        for layer in reversed(self.layers):
            beginning, end = min(layer.receive), min(layer.receive) + len(layer.output_shape)
            input_slice: TensorFlow = tuple(volume[beginning:end])
            del volume[beginning:end]
            output = layer.backprop(input_slice)
            params |= output[1]
            losses: list[tuple[int, Tensor]] = [(el, output[0][idx]) for idx, el in enumerate(layer.receive)]
            losses = sorted(losses, key=lambda x: x[0])
            for pos, loss in losses:
                volume.insert(pos, loss)
        return tuple(volume), params

    def get_data(self: Block) -> SaveData:
        data = super().get_data()
        layers_data = []
        for layer in self.layers:
            layer_data = layer.get_data()
            layer_data["class"] = layer.__class__.__name__
            layers_data.append(layer_data)
        data["layers"] = layers_data
        return data

    def load_from_data(self: Block, data: SaveData, layer_types: dict[str, type[Layer]] = {}) -> None:
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

    def get_params_data(self: Block) -> list[dict[str, Any]]:
        data = super().get_params_data()
        if len(data[0]["parameters"]) == 0:
            data = []
        for layer in self.layers:
            if isinstance(layer, Block):
                data += layer.get_params_data()
            else:
                layer_data = layer.get_params_data()
                if len(layer_data[0]["parameters"]) > 0:
                    data += layer_data
        return data

    def get_layer_by_id(self: Block, id: int) -> Layer | None:
        for layer in self.layers:
            if isinstance(layer, Block):
                nested = layer.get_layer_by_id(id)
                if nested is not None:
                    return nested
            else:
                if layer._id == id:
                    return layer
        return None

    def get_parameters(self: Block) -> int:
        nbr = super().get_parameters()
        for layer in self.layers:
            nbr += layer.get_parameters()
        return nbr

    def frozen(self: Block) -> None:
        super().freeze()
        for layer in self.layers:
            layer.freeze()

    def __getitem__(self: Block, key):
        if isinstance(key, (int, slice)):
            return self.layers[key]

        if isinstance(key, type) and issubclass(key, Layer):
            layers = []
            for layer in self.layers:
                if isinstance(layer, key):
                    layers.append(layer)
                if isinstance(layer, Block):
                    layers.extend(layer[key])  # type: ignore
            return layers

        raise TypeError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key, value):
        self.layers[key] = value

    def __delitem__(self, key):
        del self.layers[key]
