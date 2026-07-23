from core import LLaMA, Embedding
from data import scrap_text
from graphics import chat

sample = scrap_text(100_000)


def build_data() -> list[tuple[list[str], list[str]]]:
    words = sample.split(" ")
    samples: list[list[str]] = []
    lengths = [4, 8, 16, 32, 64, 128]
    for i in range(0, len(words), sum(lengths) + len(lengths)):
        idx = i
        for length in lengths:
            samples.append(words[idx : idx + length + 1])
            idx += length + 1
    data: list[tuple[list[str], list[str]]] = []
    for example in samples:
        data.append((example[:-1], example[1:]))
    return data


def llama():
    gpt_name = "llama"
    head_numbers = [8, 8, 8, 8, 8, 8, 8, 8]

    embedding = Embedding(96)  # Must be divisible by 16 (which is 2 * H)
    embedding_name = "embedding"
    if not embedding.load(embedding_name):
        embedding.build_vocab(sample)
        embedding.set_lr(1)
        embedding.cbow_training(sample, window=5, batch=1)
        embedding.save(embedding_name)

    lr = 0.001
    gpt = LLaMA(
        head_numbers=head_numbers,
        embedding=embedding,
        lr=lr,
    )
    gpt.load(gpt_name)
    gpt.set_lr(lr)

    data = build_data()
    batch = 50
    gpt.train_words(data, batch)
    gpt.save(gpt_name)

    chat(gpt)
