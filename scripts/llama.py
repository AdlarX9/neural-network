from core import LLaMA, Embedding, ByteTokenizer, WordTokenizer, Data, Trainer, LogLoss
from data import scrap_text, SaveHandler
from graphics import chat
import math


def llama():
    gpt_name = "llama-3m"
    if not SaveHandler().has(gpt_name):
        # Build Embedding
        tokenizer = ByteTokenizer(8192)
        # tokenizer = WordTokenizer()
        embedding = Embedding(tokenizer, 128)
        text = (
            scrap_text(1_000_000, filename="scrapped-0") + " " + scrap_text(1_000_000, filename="scrapped-1")
        )
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
        )
        gpt.save(gpt_name)
    else:
        gpt = LLaMA()
        gpt.load(gpt_name)

    scrap_number: int = 0
    displayed: bool = False
    data = Data()

    trainer = Trainer((gpt,), data=data, loss=LogLoss())
    trainer.adam = True
    trainer.max_lr = (5e-4,)
    trainer.final_lr = (5e-5,)
    trainer.warmup_steps = 1 * 500
    trainer.cosine_decay = (10 * 500, 45 * 500)

    try:
        while True:
            # Build data
            text = scrap_text(1_000_000, filename="scrapped-" + str(scrap_number))
            samples = data.get_samples(gpt, text)
            del text

            # Train & Save
            def save():
                gpt.save(gpt_name)

            step = 200
            idx = 1
            nbr_of_samples = math.ceil(len(samples) / step)
            while len(samples) >= step:
                title = f"{gpt_name} training scrapped-{str(scrap_number)} n°{str(idx)}/{str(nbr_of_samples)}"
                idx += 1
                data.build_tokens_data(gpt, samples[:step])
                del samples[:step]
                trainer.train(batch=1, title=title)
                save()
            title = f"{gpt_name} training scrapped-{str(scrap_number)} n°{str(idx)}/{str(nbr_of_samples)}"
            data.build_tokens_data(gpt, samples)
            trainer.train(batch=1, title=title)
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
