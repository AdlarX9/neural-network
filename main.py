from core import (
    Flatten,
    FC,
    Conv,
    ExitLoss,
    Network,
    Layer,
    ProbaExit,
    Biais,
    ReLU,
    ConvBiais,
    Tokenizer,
    Embedding,
    Recurrent,
    LSTM,
    Decoder,
)
from data import load_mnist_data, scrap_text, SaveHandler
from graphics import view_numbers, regression
import numpy as np
from numpy.typing import NDArray
import math
import random


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


def learn_shape() -> None:
    def curve(x: float, y: float) -> float:
        circle = math.sqrt(0.5 * (x - 0.5) ** 2 + (y - 0.5) ** 2)
        donut = math.sin(7.5 * circle)
        return donut

    def get_data(dim: int) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        data = []
        for _ in range(dim):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            data.append((np.array([[x], [y]]), np.array([[curve(x, y)]])))
        return data

    layers: list[Layer] = [
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(1),
        Biais(),
    ]
    network = Network(layers=layers, exit_loss=ExitLoss(), input_shape=(2, 1), lr=0.0001)

    save_handler = SaveHandler()
    name = "reproduce_shape"
    if save_handler.has(name):
        network = save_handler.load(name)
        if not isinstance(network, Network):
            raise MemoryError

    batch = 1000
    data = get_data(batch)
    network.train(data=data, batch=batch)
    save_handler.save(network, name)
    regression(network, curve)


def get_embedding(name: str) -> tuple[Embedding, bool]:
    tokenizer = Tokenizer()
    new_one = False

    # Handle save
    save_handler = SaveHandler()
    if save_handler.has(name):
        embedding = save_handler.load(name)
        if not isinstance(embedding, Embedding):
            raise MemoryError
        return embedding, new_one
    else:
        # Build Tokenizer vocab
        text_for_vocab = scrap_text(15)
        tokenizer.build_vocab(text_for_vocab)

        # Build default Embedding
        embedding = Embedding(2)
        embedding.set_input_shape((tokenizer.length(), 1))
        save_handler.save(embedding, name)
        new_one = True
        return embedding, new_one


embedding_name = "embedding"


def train_embedding() -> None:
    embedding, _ = get_embedding(embedding_name)

    def train():
        embedding.set_lr(0.1)
        text = scrap_text(15)
        embedding.cbow_training(text, window=2, batch=100_000)
        SaveHandler().save(embedding, embedding_name)

    train()


def predict_words() -> None:
    sentence: str = str("roi duc duchesse")
    number: int = 12

    # Build network
    context = 3
    embedding, new = get_embedding(embedding_name)

    if new:
        train_embedding()

    save_handler = SaveHandler()
    lstm_name = "lstm"
    if save_handler.has(lstm_name):
        network = save_handler.load(lstm_name)
        if not isinstance(network, Network):
            raise MemoryError
    else:
        layers: list[Layer] = [Recurrent(LSTM()), Decoder(embedding)]
        network = Network(
            layers=layers, exit_loss=ProbaExit(), input_shape=(embedding.output_shape[0], context), lr=0.001
        )
        save_handler.save(network, lstm_name)
    tokenizer = embedding.tokenizer

    def build_data() -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        size = 15
        text = scrap_text(size)
        words = text.split(" ")
        data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for i in range(size - context):
            entry = None
            for j in range(context):
                word = words[i + j]
                one_hot = tokenizer.get_one_hot(word).reshape(-1, 1)
                if entry is None:
                    entry = one_hot
                else:
                    entry = np.hstack((entry, one_hot))
            answer = tokenizer.get_one_hot(words[i + context]).reshape(-1, 1)
            if entry is None:
                raise ValueError
            entry = embedding.feed_forward(entry)
            data.append((entry, answer))
        return data

    def train() -> None:
        data = build_data()
        network.train(data=data, batch=100_000)

    train()
    save_handler.save(network, lstm_name)

    words = sentence.split(" ")
    if len(words) != context:
        raise ValueError("mismatch len(words) and context size:", len(words), context)

    predictions: list[str] = []

    def predict_next_word(beginning: list[str]) -> str:
        one_hot_sentence = None
        for word in beginning:
            if one_hot_sentence is None:
                one_hot_sentence = tokenizer.get_one_hot(word).reshape(-1, 1)
            else:
                one_hot_sentence = np.hstack((one_hot_sentence, tokenizer.get_one_hot(word).reshape(-1, 1)))
        if one_hot_sentence is None:
            raise ValueError
        embedded_sentence = embedding.feed_forward(one_hot_sentence)
        one_hot_prediction = network.compute(embedded_sentence)
        prediction = tokenizer.get_word(int(np.argmax(one_hot_prediction)))
        return prediction

    for _ in range(number):
        prediction = predict_next_word(words)
        words.append(prediction)
        words.pop(0)
        predictions.append(prediction)

    print(sentence, "|", " ".join(predictions))


def main() -> None:
    predict_words()


if __name__ == "__main__":
    main()
