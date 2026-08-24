from data import SaveHandler, load_mnist_data
from core import DDPM2M, Trainer, WordTokenizer, Data
from graphics import image_generator
from math import ceil


def image_generation() -> None:
    ddpm_name = "image_generation_2"
    if not SaveHandler().has(ddpm_name):
        tokenizer = WordTokenizer()
        tokenizer.forced = True
        tokenizer.build_vocab("0 1 2 3 4 5 6 7 8 9")
        ddpm = DDPM2M(
            L=1,
            T=200,
            tokenizer=tokenizer,
        )
    else:
        ddpm = DDPM2M()
        ddpm.load(ddpm_name)

    config = [
        {
            "layer": ddpm,
            "adam": True,
            "max_lr": 3e-4,
            "final_lr": 3e-5,
            "cosine_decay": (0, 50_000),
        }
    ]
    data = Data()
    trainer = Trainer(config, data)

    samples = load_mnist_data()
    samples = [(text, image) for image, text in samples]

    step = 500
    idx: int = 1
    nbr_of_samples = ceil(len(samples) / step)

    def train(size: int = step):
        title = f"{ddpm_name} training n°{str(idx)}/{str(nbr_of_samples)}"
        data.build_ddpm_data(ddpm, samples[:size])
        del samples[:size]
        trainer.train(batch=1, title=title)
        ddpm.save(ddpm_name)

    displayed: bool = False
    try:
        while len(samples) >= step:
            train(step)
            idx += 1
        train(len(samples))
    except KeyboardInterrupt:
        displayed = True
        image_generator(ddpm)
    finally:
        if not displayed:
            image_generator(ddpm)
