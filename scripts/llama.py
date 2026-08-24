from core import LLaMA, ByteTokenizer, Data, Trainer, LogLoss
from data import scrap_text, SaveHandler
from graphics import chat
import math


def llama():
    gpt_name = "llama-3m"
    if not SaveHandler().has(gpt_name):
        gpt = LLaMA(
            vocab_size=8192,
            embedding_dim=128,
            head_numbers=[4, 4, 4, 4, 4, 4],
        )
        megatext = (
            scrap_text(1_000_000, filename="scrapped-0") + " " + scrap_text(1_000_000, filename="scrapped-1")
        )
        gpt.build_vocab(megatext)
        del megatext
        gpt.save(gpt_name)
    else:
        gpt = LLaMA()
        gpt.load(gpt_name)

    units = [
        {
            "layer": gpt,
            "adam": True,
            "max_lr": 5e-4,
            "final_lr": 5e-5,
            "warmup_steps": 1 * 500,
            "cosine_decay": (10 * 500, 50 * 500),
        }
    ]
    data = Data()
    trainer = Trainer(units, data=data, loss=LogLoss())

    displayed: bool = False
    scrap_number: int = 0
    step = 200
    try:
        while True:
            text = scrap_text(1_000_000, filename="scrapped-" + str(scrap_number))
            samples = data.get_samples(gpt, text)
            del text

            idx = 1
            nbr_of_samples = math.ceil(len(samples) / step)
            while len(samples) >= step:
                title = f"{gpt_name} training scrapped-{str(scrap_number)} n°{str(idx)}/{str(nbr_of_samples)}"
                idx += 1
                data.build_tokens_data(gpt, samples[:step])
                del samples[:step]
                trainer.train(batch=1, title=title)
                gpt.save(gpt_name)

            title = f"{gpt_name} training scrapped-{str(scrap_number)} n°{str(idx)}/{str(nbr_of_samples)}"
            data.build_tokens_data(gpt, samples)
            trainer.train(batch=1, title=title)
            gpt.save(gpt_name)
            if isinstance(gpt.embedding.tokenizer, ByteTokenizer):
                scrap_number += 1

    except KeyboardInterrupt:
        # Chat
        displayed = True
        chat(gpt)
    finally:
        if not displayed:
            chat(gpt)
