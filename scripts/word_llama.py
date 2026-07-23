from core import LLaMA, TextNetwork, Embedding, WordTokenizer
from .word_lstm import sample
from graphics import chat


def build_data(gpt: TextNetwork) -> list[tuple[list[int], list[int]]]:
    tokens = gpt.tokenize(sample)
    data = [(tokens[:-1], tokens[1:])]
    return data


def word_llama():
    gpt_name = "word_llama"
    head_numbers = [1]

    embedding = Embedding(WordTokenizer(), 4)
    embedding_name = "word_llama_embedding"
    if not embedding.load(embedding_name):
        embedding.build_vocab(sample)
        embedding.set_lr(0.1)
        embedding.cbow_training(sample, window=2, batch=4_000)
        embedding.save(embedding_name)

    lr = 0.01
    gpt = LLaMA(
        head_numbers=head_numbers,
        embedding=embedding,
        lr=lr,
    )
    gpt.load(gpt_name)
    gpt.set_lr(lr)

    data = build_data(gpt)
    batch = 4_000
    gpt.train_tokens(data, batch)
    gpt.save(gpt_name)

    predictions = gpt.compute_text(gpt.untokenize(gpt.tokenize(sample)[:-1]))
    print("Trained on:")
    print(sample)
    print("and learned:")
    print(gpt.untokenize(gpt.tokenize(sample)[:1]) + " |" + predictions)
