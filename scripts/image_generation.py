from data import SaveHandler, load_mnist_data
from core import DDPM, Trainer, WordTokenizer, Data


def image_generation() -> None:
    ddpm_name = "image_generation"
    if not SaveHandler().has(ddpm_name):
        tokenizer = WordTokenizer()
        tokenizer.build_vocab("0 1 2 3 4 5 6 7 8 9", force=True)
        ddpm = DDPM(
            L=1,
            T=1000,
            head_numbers=[],
            tokenizer=tokenizer,
        )
    else:
        ddpm = DDPM()
        ddpm.load(ddpm_name)

    config = [
        {
            "layer": ddpm,
            "max_lr": 1e-3,
            "adam": True,
        }
    ]
    samples = load_mnist_data()[:100]
    samples = [(text, image) for image, text in samples]
    data = Data()
    data.build_ddpm_data(ddpm, samples)
    trainer = Trainer(config, data)
    trainer.train(batch=1)
    ddpm.save(ddpm_name)
