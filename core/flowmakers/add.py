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

        def broadcast_shapes_manual(*shapes: tuple[int, ...]) -> tuple[int, ...]:
            """
            Broadcasting NumPy avec support des dimensions inconnues (-1).

            Règles :
                a + a  -> a
                1 + a  -> a
                -1 + -1 -> -1
                -1 + a  -> -1

            Les shapes sont alignées à droite comme avec NumPy.
            """

            max_ndim = max(len(shape) for shape in shapes)

            # On aligne toutes les shapes à droite.
            aligned_shapes = [(1,) * (max_ndim - len(shape)) + shape for shape in shapes]

            output_shape = []

            for dimensions in zip(*aligned_shapes):

                known = [dim for dim in dimensions if dim != -1]

                # Toutes les dimensions sont inconnues.
                if not known:
                    output_shape.append(-1)
                    continue

                # Les dimensions connues doivent être compatibles.
                non_one = [dim for dim in known if dim != 1]

                if len(set(non_one)) > 1:
                    raise ValueError(f"Incompatible shapes for broadcasting: {shapes}")

                # S'il y a au moins un -1, on ne connaît pas
                # suffisamment la dimension finale.
                if -1 in dimensions:
                    output_shape.append(-1)
                else:
                    # Toutes les dimensions sont connues :
                    # 1 est broadcasté vers la dimension non-1.
                    output_shape.append(non_one[0] if non_one else 1)

            return tuple(output_shape)

        # Vérification / calcul du broadcasting.
        try:
            output_shape = broadcast_shapes_manual(*input_shape)
        except ValueError:
            raise ValueError(f"Incompatible shapes for broadcasting: {input_shape}")

        self.output_shape = (output_shape,)
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
