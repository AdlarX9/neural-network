from core import LSTMNetwork, ProbaExit, Embedding
import numpy as np
from numpy.typing import NDArray

sample: str = str(
    "roi duc duchesse prince princesse bisous amour mariage lit dormir repos travail etat salaire argent"
)


def build_data(context: int = 1) -> list[tuple[list[str], list[str]]]:
    text = sample
    words = text.split(" ")
    data: list[tuple[list[str], list[str]]] = []
    for i in range(len(words) - context):
        entry = words[i : i + context]
        answer = [words[i + context]]
        data.append((entry, answer))
    return data


def word_prediction() -> None:
    # Build Embedding
    embedding = Embedding(4)
    embedding_name = "embedding"
    if not embedding.load(embedding_name):
        embedding.build_vocab(sample)
        embedding.set_lr(0.1)
        embedding.cbow_training(sample, window=2, batch=40_000)
        embedding.save(embedding_name)

    # Build LSTM
    context = 3
    lstm_name = "lstm"
    lstm = LSTMNetwork(embedding=embedding, exit_loss=ProbaExit(), lr=0.05)
    lstm.load(lstm_name)
    data = build_data(context)
    lstm.train_words(data=data, batch=2_000)
    lstm.save(lstm_name)

    # Predict words
    number: int = 12
    words = sample.split(" ")[:context]
    predictions: list[str] = []
    for _ in range(number):
        prediction = lstm.predict_next_word(words)
        words.append(prediction)
        words.pop(0)
        predictions.append(prediction)

    print("Trained on:")
    print(sample)
    print("And generated:")
    print(" ".join(sample.split(" ")[:context]), "|", " ".join(predictions))
