from core import LLaMA, TextNetwork, Embedding, WordTokenizer, Data, Trainer, LogLoss, Tokens
from .word_lstm import sample


def build_data(gpt: TextNetwork) -> list[tuple[Tokens, Tokens]]:
    tokens = gpt.tokenize(sample)
    data = [(tokens[:-1], tokens[1:])]
    return data


def word_llama():
    gpt_name = "word_llama"
    head_numbers = [6]

    embedding = Embedding(WordTokenizer(), 12)
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

    tokens_data = build_data(gpt)
    data = Data()
    data.build_tokens_data(gpt, tokens_data)
    batch = 4_000
    trainer = Trainer(data)
    trainer.train((gpt,), loss=LogLoss(), batch=batch)
    gpt.save(gpt_name)

    predictions = gpt.compute_text(gpt.untokenize(gpt.tokenize(sample)[:-1]))
    print("Trained on:")
    print(sample)
    print("and learned:")
    print(gpt.untokenize(gpt.tokenize(sample)[:1]), "|", predictions)
