from __future__ import annotations
from ..basics.layer import Layer
from ..utils.typing import ShapeFlow, Tensor, TensorFlow, Receive2


class Matmul(Layer):
    def __init__(self: Matmul, receive: Receive2 = (0, 1)) -> None:
        self._receive = 2
        super().__init__(receive)

    def set_input_shape(self: Matmul, input_shape: ShapeFlow) -> ShapeFlow:
        super().set_input_shape(input_shape)
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        C, n, p = None, None, None
        D, q, r = None, None, None
        if len(shape1) == 2:
            n, p = shape1
        else:
            C, n, p = shape1
        if len(shape2) == 2:
            q, r = shape2
        else:
            D, q, r = shape2
        if q != p:
            raise ValueError("Incompatible dimension for matrix multiplication:", shape1, shape2)
        if C is None:
            if D is None:
                self.output_shape = ((n, r),)
            else:
                self.output_shape = ((D, n, r),)
        else:
            self.output_shape = ((C, n, r),)
            if D is not None and D != C:
                raise ValueError
        return self.output_shape

    def feed_forward(
        self: Matmul,
        entry: TensorFlow,
    ) -> Tensor:
        return entry[0] @ entry[1]

    def descend_gradient(
        self: Matmul,
        gradient: Tensor,
    ) -> TensorFlow:
        if self.input is None:
            raise MemoryError
        l0 = len(self.input[0].shape)
        l1 = len(self.input[1].shape)
        gradients = (
            gradient @ self.input[1].swapaxes(l1 - 2, l1 - 1),
            self.input[0].swapaxes(l0 - 2, l0 - 1) @ gradient,
        )
        return gradients
