from core import TextNetwork, LSTMNetwork, ProbaExit, Embedding, WordTokenizer

sample: str = str(
    "roi duc duchesse prince princesse bisous amour mariage lit dormir repos travail etat salaire argent"
)


def build_data(lstm: TextNetwork, context: int = 1) -> list[tuple[list[int], list[int]]]:
    tokens = lstm.tokenize(sample)
    data: list[tuple[list[int], list[int]]] = []
    for i in range(len(tokens) - context):
        entry = tokens[i : i + context]
        answer = [tokens[i + context]]
        data.append((entry, answer))
    return data


def word_lstm() -> None:
    # Build Embedding
    embedding: Embedding = Embedding(tokenizer=WordTokenizer(), dim=4)
    embedding_name = "word_lstm_embedding"
    if not embedding.load(embedding_name):
        embedding.build_vocab(sample)
        embedding.set_lr(0.1)
        embedding.cbow_training(sample, window=2, batch=4_000)
        embedding.save(embedding_name)

    # Build LSTM
    context = 3
    lstm_name = "word_lstm"
    lstm = LSTMNetwork(embedding=embedding, exit_loss=ProbaExit(), lr=0.05)
    lstm.load(lstm_name)
    data = build_data(lstm, context)
    lstm.train_tokens(data=data, batch=2_000)
    lstm.save(lstm_name)

    # Predict words
    number: int = 12
    tokens: list[int] = lstm.tokenize(sample)[:context]
    original_tokens = tokens.copy()
    predictions: list[int] = []
    for _ in range(number):
        prediction = lstm.predict_next_token(lstm.untokenize(tokens))
        tokenized_prediction = lstm.tokenize(prediction)
        tokens += tokenized_prediction
        tokens.pop(0)
        predictions += tokenized_prediction

    print("Trained on:")
    print(sample)
    print("And generated:")
    print(lstm.untokenize(original_tokens), "|", lstm.untokenize(predictions))
