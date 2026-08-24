from __future__ import annotations

from ..utils.typing import Receive
from ..basics.block import Block
from ..basics.layer import Layer
from ..flowmakers.duplicate import Duplicate


class UNet(Block):
    def __init__(
        self: UNet,
        down_layers: list[Layer] = [],
        up_layers: list[Layer] = [],
        receive: Receive = (0,),
    ) -> None:
        state: dict[int, int] = {idx: idx for idx in range(len(receive))}

        def move_down_from(idx: int) -> None:
            for key in range(idx, len(receive)):
                state[key] += 1

        def move_up_from(idx: int) -> None:
            for key in range(idx, len(receive)):
                state[key] -= 1

        def get_receive(receive: Receive) -> Receive:
            return tuple([state[idx] for idx in receive])

        layers: list[Layer] = []

        for layer in down_layers:
            logical_receive = layer.receive
            for val in logical_receive:
                layers.append(Duplicate(factor=2, receive=(state[val],)))
                move_down_from(val)
            layer.receive = get_receive(logical_receive)
            layers.append(layer)
            for val in logical_receive:
                if val != 0:
                    move_up_from(val)
        for layer in up_layers:
            logical_receive = layer.receive
            for val in logical_receive:
                if val != 0:
                    layers.append(Duplicate(factor=2, receive=(state[val],)))
                    move_down_from(val)
            layer.receive = (state[0] - 1,) + get_receive(logical_receive)
            layers.append(layer)
            for val in logical_receive:
                move_up_from(val)

        super().__init__(layers, receive)
