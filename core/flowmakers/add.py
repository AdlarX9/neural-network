from __future__ import annotations
import numpy as np
from ..basics.layer import Layer
from ..utils.typing import ShapeFlow, Tensor, TensorFlow, Receive


class Add(Layer):
    def __init__(self: Add, receive: Receive = (0,)) -> None:
        super().__init__(receive)

    def set_input_shape(self: Add, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)

        if len(input_shape) == 0:
            raise ValueError("Add requires at least one input")

        # Vérifie que les shapes sont compatibles avec le broadcasting NumPy
        reference_shape = input_shape[0]

        for shape in input_shape[1:]:
            try:
                np.broadcast_shapes(reference_shape, shape)
            except ValueError:
                raise ValueError(
                    "Incompatible shapes for broadcasting:",
                    reference_shape,
                    shape,
                )

        self.output_shape = (np.broadcast_shapes(*input_shape),)

        return self.output_shape

    def feed_forward(
        self: Add,
        entry: TensorFlow,
    ) -> Tensor:
        output = entry[0].copy()

        for i in range(1, len(entry)):
            output += entry[i]

        return output

    def descend_gradient(
        self: Add,
        gradient: Tensor,
    ) -> TensorFlow:
        if self.input is None:
            raise MemoryError

        return tuple(self._unbroadcast(gradient, x.shape) for x in self.input)

    @staticmethod
    def _unbroadcast(
        gradient: Tensor,
        target_shape: tuple,
    ) -> Tensor:
        result = gradient

        # Ajouter des dimensions à gauche jusqu'à avoir
        # le même nombre de dimensions
        while result.ndim > len(target_shape):
            result = np.sum(result, axis=0)

        # Pour chaque dimension broadcastée (= 1),
        # on somme le gradient sur cet axe.
        for axis, size in enumerate(target_shape):
            if size == 1 and result.shape[axis] != 1:
                result = np.sum(
                    result,
                    axis=axis,
                    keepdims=True,
                )

        return result.reshape(target_shape)
