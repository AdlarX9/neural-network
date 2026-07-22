from core import CNN, ProbaExit
from data import load_mnist_data
from graphics import view_numbers


def number_recognition() -> None:
    name = "number_recognition"
    cnn = CNN(
        parameters=([(1, 28, 28, 0), (32, 3, 2, -1), (64, 3, 2, -1), (128, 3, 2, -1)], [10]),
        exit_loss=ProbaExit(),
        lr=0.005,
    )
    cnn.load(name)

    data = load_mnist_data()
    cnn.train(data=data[:60000], batch=1)
    cnn.save(name)

    view_numbers(cnn)
