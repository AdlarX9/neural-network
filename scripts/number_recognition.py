from core import Network, CNN, ProbaExit
from data import load_mnist_data, SaveHandler
from graphics import view_numbers


def number_recognition() -> None:
    cnn = CNN(
        parameters=([(1, 28, 28, 0), (32, 3, 2, -1), (64, 3, 2, -1), (128, 3, 2, -1)], [10]), exit_loss=ProbaExit(), lr=0.005
    )

    save_handler = SaveHandler()
    name = "number_recognition"
    if save_handler.has(name):
        cnn = save_handler.load(name)
        if not isinstance(cnn, Network):
            raise MemoryError

    data = load_mnist_data()
    cnn.train(data=data, batch=1)
    save_handler.save(cnn, name)

    view_numbers(cnn)
