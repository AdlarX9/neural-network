from core import LLaMA, Embedding, ByteTokenizer
from graphics import ConsoleVisualization
from data import scrap_text, SaveHandler
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
    gpt_name = "llama"
    lr = 0.0008
    if not SaveHandler().has(gpt_name):
        # Build Embedding
        tokenizer = ByteTokenizer(8192)
        embedding = Embedding(tokenizer, 96)
        text = scrap_text(1_000_000, filename="scrapped-0")
        embedding.build_vocab(text)
        embedding.set_input_shape((tokenizer.length(), -1))

        # Build LLaMA
        head_numbers = [4, 4, 4, 4, 4, 4]
        gpt = LLaMA(
            head_numbers=head_numbers,
            embedding=embedding,
            lr=lr,
        )
    else:
        gpt = LLaMA()
        gpt.load(gpt_name)
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

            visualization = ConsoleVisualization()
            step = 500
            idx = 0
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
