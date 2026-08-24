from data import SaveHandler, load_mnist_data
from core import Diffusion2M, Trainer, WordTokenizer, Data
from graphics import image_generator
from math import ceil


def image_generation() -> None:
    diffusion_name = "image_generation"
    if not SaveHandler().has(diffusion_name):
        tokenizer = WordTokenizer()
        tokenizer.forced = True
        tokenizer.build_vocab("0 1 2 3 4 5 6 7 8 9")
        diffusion = Diffusion2M(
            L=1,
            T=200,
            tokenizer=tokenizer,
        )
    else:
        diffusion = Diffusion2M()
        diffusion.load(diffusion_name)
    diffusion.jump = 15

    config = [
        {
            "layer": diffusion,
            "adam": True,
            "max_lr": 3e-4,
            "final_lr": 3e-5,
            "cosine_decay": (0, 300_000),
        }
    ]
    data = Data()
    trainer = Trainer(config, data)

    samples = load_mnist_data()
    samples = [(text, image) for image, text in samples]

    step = 1000
    idx: int = 1
    nbr_of_samples = ceil(len(samples) / step)

    def train(size: int = step):
        title = f"{diffusion_name} training n°{str(idx)}/{str(nbr_of_samples)}"
        data.build_diffusion_data(diffusion, samples[:size])
        del samples[:size]
        trainer.train(batch=1, title=title)
        diffusion.save(diffusion_name)

    displayed: bool = False
    try:
        while True:
            while len(samples) >= step:
                train(step)
                idx += 1
            train(len(samples))
    except KeyboardInterrupt:
        displayed = True
        image_generator(diffusion, "ddim")
    finally:
        if not displayed:
            image_generator(diffusion, "ddim")
