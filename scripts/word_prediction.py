from core import (
    Network,
    Layer,
    ProbaExit,
    Tokenizer,
    Embedding,
    Recurrent,
    LSTM,
    Decoder,
)
from data import SaveHandler
import numpy as np
from numpy.typing import NDArray

sample: str = str("roi duc duchesse prince princesse bisous amour mariage lit dormir repos travail etat salaire argent")
embedding_name: str = str("embedding")
lstm_name: str = str("lstm")
sentence: str = str("roi duc duchesse")


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
        text_for_vocab = sample
        tokenizer.build_vocab(text_for_vocab)

        # Build default Embedding
        embedding = Embedding(3)
        embedding.set_input_shape((tokenizer.length(), -1))
        save_handler.save(embedding, name)
        new_one = True
        return embedding, new_one


def train_embedding(embedding: Embedding) -> None:
    embedding.set_lr(0.1)
    text = sample
    embedding.cbow_training(text, window=2, batch=400_000)
    SaveHandler().save(embedding, embedding_name)


def build_data(embedding: Embedding) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    context = 4
    text = sample
    words = text.split(" ")
    data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for i in range(len(words) - context):
        entry = None
        for j in range(context):
            word = words[i + j]
            one_hot = embedding.tokenizer.get_one_hot(word).reshape(-1, 1)
            if entry is None:
                entry = one_hot
            else:
                entry = np.hstack((entry, one_hot))
        answer = embedding.tokenizer.get_one_hot(words[i + context]).reshape(-1, 1)
        if entry is None:
            raise ValueError
        data.append((entry, answer))
    return data


def predict_next_word(beginning: list[str], embedding: Embedding, lstm: Network) -> str:
    one_hot_sentence = None
    for word in beginning:
        if one_hot_sentence is None:
            one_hot_sentence = embedding.tokenizer.get_one_hot(word).reshape(-1, 1)
        else:
            one_hot_sentence = np.hstack(
                (one_hot_sentence, embedding.tokenizer.get_one_hot(word).reshape(-1, 1))
            )
    if one_hot_sentence is None:
        raise ValueError
    for layer in lstm.layers:
        if isinstance(layer, Recurrent):
            layer.reset_data()
    one_hot_prediction = lstm.compute(one_hot_sentence)
    prediction = embedding.tokenizer.get_word(int(np.argmax(one_hot_prediction)))
    return prediction


def get_lstm(embedding: Embedding) -> Network:
    save_handler = SaveHandler()
    if save_handler.has(lstm_name):
        network = save_handler.load(lstm_name)
        if not isinstance(network, Network):
            raise MemoryError
        return network
    else:
        layers: list[Layer] = [embedding, Recurrent(LSTM()), Decoder(embedding)]
        network = Network(
            layers=layers, exit_loss=ProbaExit(), input_shape=(embedding.input_shape[0], -1), lr=0.05
        )
        save_handler.save(network, lstm_name)
        return network


def word_prediction() -> None:
    # Build Embedding
    embedding, new = get_embedding(embedding_name)
    if new:
        train_embedding(embedding)

	# Build LSTM
    save_handler = SaveHandler()
    lstm = get_lstm(embedding)
    data = build_data(embedding)
    lstm.train(data=data, batch=30_000)
    save_handler.save(lstm, lstm_name)

    # Predict words
    number: int = 12
    words = sentence.split(" ")
    predictions: list[str] = []
    for _ in range(number):
        prediction = predict_next_word(words, embedding, lstm)
        words.append(prediction)
        words.pop(0)
        predictions.append(prediction)
    print(sentence, "|", " ".join(predictions))
