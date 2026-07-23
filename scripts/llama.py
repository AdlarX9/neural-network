from core import LLaMA, Embedding, ByteTokenizer, WordTokenizer
from data import scrap_text
from graphics import chat

text = scrap_text(100_000)


def build_data(tokens: list[int]) -> list[tuple[list[int], list[int]]]:
    samples: list[list[int]] = []
    lengths = [4, 8, 16, 32, 64, 128]
    for i in range(0, len(tokens), sum(lengths) + len(lengths)):
        idx = i
        for length in lengths:
            samples.append(tokens[idx : idx + length + 1])
            idx += length + 1
    data: list[tuple[list[int], list[int]]] = []
    for example in samples:
        if len(example) >= 2:
            data.append((example[:-1], example[1:]))
    return data


def llama():
    # Cache
    tokens: list[int] | None = None

    # Build Embedding
    embedding = Embedding(ByteTokenizer(8_192), 96)  # Must be divisible by 16 (which is 2 * H)
    embedding_name = "llama_embedding"
    if not embedding.load(embedding_name):
        embedding.build_vocab(text)
        embedding.set_lr(1)
        tokens = embedding.tokenizer.tokenize(text)
        embedding.cbow_training(tokens, window=5, batch=1)
        embedding.save(embedding_name)

    # Build LLaMA
    gpt_name = "llama"
    head_numbers = [8, 8, 8, 8, 8, 8, 8, 8]
    lr = 0.001
    gpt = LLaMA(
        head_numbers=head_numbers,
        embedding=embedding,
        lr=lr,
    )
    gpt.load(gpt_name)
    gpt.set_lr(lr)
    gpt.embedding = embedding

    # Build data
    if tokens is None:
        tokens = embedding.tokenizer.tokenize(text)
    data = build_data(tokens)

    # Train
    batch = 50
    gpt.train_tokens(data, batch)

    # Save
    gpt.embedding.save(embedding_name)
    gpt.save(gpt_name)

    # Chat
    chat(gpt)
