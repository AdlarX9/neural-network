from core import LLaMA, Embedding, ByteTokenizer
from graphics import ConsoleVisualization
from data import scrap_text
from graphics import chat
import random


def build_data(
    tokens: list[int],
    context_length: int = 256,
    stride: int = 128,
) -> list[tuple[list[int], list[int]]]:
    data: list[tuple[list[int], list[int]]] = []
    for i in range(0, len(tokens) - context_length, stride):
        data.append((tokens[i : i + context_length], tokens[i + 1 : i + context_length + 1]))
    return data


def llama():
    # Build Embedding
    tokenizer = ByteTokenizer()
    embedding = Embedding(tokenizer, 256)
    embedding_name = "llama_embedding"
    if not embedding.load(embedding_name):
        megatext = scrap_text(1_000_000, filename="scrapped-0")
        embedding.build_vocab(megatext)
        embedding.set_input_shape((tokenizer.length(), -1))
        embedding.set_lr(1)
        megatokens = embedding.tokenize(megatext)
        del megatext
        embedding.cbow_training(text=megatokens[:100_000], window=8, batch=1)
        embedding.save(embedding_name)
        del megatokens

    # Build LLaMA
    gpt_name = "llama"
    head_numbers = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    lr = 0.0001
    gpt = LLaMA(
        head_numbers=head_numbers,
        embedding=embedding,
        lr=lr,
    )
    gpt.load(gpt_name)
    gpt.embedding = embedding
    gpt.set_lr(lr)

    scrap_number = 0
    try:
        while True:
            # Build data
            text = scrap_text(1_000_000, filename="scrapped-" + str(scrap_number))
            scrap_number += 1
            tokens = gpt.tokenize(text)
            del text
            data = build_data(tokens)
            random.shuffle(data)
            del tokens

            # Train & Save
            def save():
                gpt.save(gpt_name)
                gpt.embedding.save(embedding_name)

            visualization = ConsoleVisualization()
            step = 200
            idx = 1
            nbr_of_samples = len(data) // step
            while len(data) >= step:
                visualization.title = "LLaMA Training Sample n°" + str(idx) + "/" + str(nbr_of_samples)
                idx += 1
                gpt.train_tokens(
                    data=data[:step],
                    batch=1,
                    visualization=visualization,
                )
                save()
                del data[:step]
            visualization.title = "LLaMA Training Sample n°" + str(idx) + "/" + str(nbr_of_samples)
            gpt.train_tokens(
                data=data,
                batch=1,
                visualization=visualization,
            )
            save()
    except KeyboardInterrupt:
        # Chat
        chat(gpt)
    finally:
        chat(gpt)
