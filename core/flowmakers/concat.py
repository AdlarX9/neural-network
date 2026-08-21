from __future__ import annotations
from ..basics.layer import Layer
import numpy as np
from ..utils.typing import ShapeFlow, Tensor, TensorFlow, Receive, SaveData


class Concat(Layer):
    def __init__(self: Concat, axis: int = 0, receive: Receive = (0,)) -> None:
        self._receive = -1
        self.axis = axis
        super().__init__(receive)

    def _normalize_axis(self: Concat, ndim: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("Concat axis out of range:", self.axis, ndim)
        return axis

    def set_input_shape(self: Concat, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        if len(input_shape) == 0:
            raise ValueError("Concat requires at least one input")
        reference_shape = input_shape[0]
        rank = len(reference_shape)
        axis = self._normalize_axis(rank)
        for shape in input_shape[1:]:
            if len(shape) != rank:
                raise ValueError("Concat incompatible rank:", shape, reference_shape)
            for idx, (dim, ref_dim) in enumerate(zip(shape, reference_shape)):
                if idx == axis:
                    continue
                if dim != ref_dim:
                    raise ValueError("Concat inconsistent dimensions:", shape, reference_shape)
        self.output_shape = (
            tuple(
                sum(shape[axis] for shape in input_shape) if idx == axis else reference_shape[idx]
                for idx in range(rank)
            ),
        )
        return self.output_shape

    def feed_forward(self: Concat, entry: TensorFlow) -> Tensor:
        return np.concatenate(entry, axis=self._normalize_axis(len(entry[0].shape)))

    def descend_gradient(self: Concat, gradient: Tensor) -> TensorFlow:
        if self.input is None:
            raise MemoryError
        axis = self._normalize_axis(len(self.input[0].shape))
        gradients: list[Tensor] = []
        start = 0
        for tensor in self.input:
            length = tensor.shape[axis]
            index = [slice(None)] * gradient.ndim
            index[axis] = slice(start, start + length)
            gradients.append(gradient[tuple(index)].copy())
            start += length
        return tuple(gradients)

    def get_data(self: Concat) -> dict:
        data = super().get_data()
        data["axis"] = self.axis
        return data

    def load_from_data(self: Concat, data: SaveData) -> None:
        super().load_from_data(data)
        self.axis = data["axis"]
