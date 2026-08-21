from core import CNN, Softmax, Trainer, LogLoss, Data
from data import load_mnist_data
from graphics import view_numbers


def number_recognition() -> None:
    name = "number_recognition"
    cnn = CNN(
        parameters=([(1, 28, 28, 0), (32, 3, 2, -1), (64, 3, 2, -1), (128, 3, 2, -1)], [10]),
        lr=0.005,
        more_layers=[Softmax()],
    )
    cnn.load(name)

    data = load_mnist_data()[:30_000]
    trainer = Trainer(Data(data))
    trainer.train((cnn,), loss=LogLoss(), batch=1)
    cnn.save(name)

    view_numbers(cnn)
