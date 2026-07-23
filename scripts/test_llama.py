from core import LLaMA, Embedding
from .word_prediction import sample


def build_data() -> list[tuple[list[str], list[str]]]:
    words = sample.split(" ")
    data = [(words[:-1], words[1:])]
    return data


def test_llama():
    gpt_name = "test_llama"
    head_numbers = [1]

    embedding = Embedding(4)
    embedding_name = "embedding"
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

    data = build_data()
    batch = 4_000
    gpt.train_words(data, batch)
    gpt.save(gpt_name)

    predictions = gpt.compute_words(sample.split(" ")[:-1])
    print("Trained on:")
    print(sample)
    print("and learned:")
    print(sample.split(" ")[0], "|", " ".join(predictions))
