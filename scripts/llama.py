from core import LLaMA, Embedding, ByteTokenizer, WordTokenizer
from graphics import ConsoleVisualization
from data import scrap_text, SaveHandler
from graphics import chat
import math


def llama():
    gpt_name = "llama"
    lr = 0.0008
    if not SaveHandler().has(gpt_name):
        # Build Embedding
        tokenizer = ByteTokenizer(8192)
        # tokenizer = WordTokenizer()
        embedding = Embedding(tokenizer, 128)
        text = scrap_text(1_000_000, filename="scrapped-0")
        embedding.build_vocab(text)
        embedding.set_input_shape(((tokenizer.length(), -1),))
        if isinstance(tokenizer, WordTokenizer):
            embedding.set_lr(0.1)
            smaller_text = scrap_text(10000, filename="scrapped-0")
            embedding.cbow_training(smaller_text, batch=1)

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

    scrap_number: int = 0
    displayed: bool = False
    try:
        while True:
            # Build data
            text = scrap_text(1_000_000, filename="scrapped-" + str(scrap_number))
            data = gpt.build_data(text)
            del text

            # Train & Save
            def save():
                gpt.save(gpt_name)

            visualization = ConsoleVisualization()
            step = 500
            idx = 1
            nbr_of_samples = math.ceil(len(data) / step)
            while len(data) >= step:
                visualization.title = (
                    "LLaMA Training Sample ("
                    + str(scrap_number)
                    + ") n°"
                    + str(idx)
                    + "/"
                    + str(nbr_of_samples)
                )
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
            if isinstance(gpt.embedding.tokenizer, ByteTokenizer):
                scrap_number += 1
    except KeyboardInterrupt:
        # Chat
        displayed = True
        chat(gpt)
    finally:
        if not displayed:
            chat(gpt)
