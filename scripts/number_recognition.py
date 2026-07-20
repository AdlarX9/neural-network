from core import (
    Flatten,
    FC,
    Conv,
    Network,
    ProbaExit,
    Biais,
    ReLU,
    ConvBiais,
)
from data import load_mnist_data, SaveHandler
from graphics import view_numbers


def number_recognition() -> None:
    layers = [
        Conv(32, 3, 2),
        ConvBiais(),
        ReLU(),
        Conv(64, 3, 2),
        ConvBiais(),
        ReLU(),
        Conv(128, 3, 2),
        ConvBiais(),
        ReLU(),
        Flatten(),
        FC(10),
        Biais(),
    ]
    network = Network(layers=layers, exit_loss=ProbaExit(), input_shape=(1, 28, 28), lr=0.005)

    save_handler = SaveHandler()
    name = "number_recognition"
    if save_handler.has(name):
        network = save_handler.load(name)
        if not isinstance(network, Network):
            raise MemoryError

    data = load_mnist_data()
    network.train(data=data, batch=1)
    save_handler.save(network, name)

    view_numbers(network)
