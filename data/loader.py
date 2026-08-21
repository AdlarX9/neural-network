import numpy as np
from numpy.typing import NDArray
import gzip
import struct
import urllib.request
from pathlib import Path
import os
from core import TrainData


def _load_image_as_input(image_path: Path, image_size: tuple[int, int]) -> NDArray[np.float64]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow est requis pour charger des images. Installe-le avec 'pip install pillow'."
        ) from exc

    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        if image_size is not None:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR", 2)
            grayscale = grayscale.resize(image_size, resampling)
        array = np.asarray(grayscale, dtype=np.float64) / 255.0

    return array.reshape(1, array.shape[0], array.shape[1])


def _one_hot(label: int, num_classes: int = 10) -> NDArray[np.float64]:
    encoded = np.zeros((num_classes, 1), dtype=np.float64)
    encoded[label, 0] = 1.0
    return encoded


def load_digit_data(
    root_dir: str | Path, image_size: tuple[int, int] = (28, 28)
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    root_path = Path(root_dir)
    data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    if not root_path.exists():
        return data

    for label_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        try:
            label = int(label_dir.name)
        except ValueError:
            continue

        for image_path in sorted(path for path in label_dir.iterdir() if path.is_file()):
            try:
                entry = _load_image_as_input(image_path, image_size)
            except OSError:
                continue
            data.append((entry, _one_hot(label)))

    return data


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    with urllib.request.urlopen(url) as response, destination.open("wb") as target_file:
        target_file.write(response.read())


def _load_mnist_images(file_path: Path) -> NDArray[np.float64]:
    with gzip.open(file_path, "rb") as gz_file:
        magic, count, rows, cols = struct.unpack(">IIII", gz_file.read(16))
        if magic != 2051:
            raise ValueError(f"Fichier image MNIST invalide: {file_path}")
        buffer = gz_file.read()

    images = np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows, cols)
    return images.astype(np.float64) / 255.0


def _load_mnist_labels(file_path: Path) -> NDArray[np.int64]:
    with gzip.open(file_path, "rb") as gz_file:
        magic, count = struct.unpack(">II", gz_file.read(8))
        if magic != 2049:
            raise ValueError(f"Fichier label MNIST invalide: {file_path}")
        buffer = gz_file.read()

    labels = np.frombuffer(buffer, dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(f"Nombre de labels MNIST incohérent dans {file_path}")
    return labels.astype(np.int64)


def load_mnist_data(
    cache_dir: str | Path = Path(os.path.join("data", "numbers")), split: str = "train"
) -> TrainData:
    cache_path = Path(cache_dir)

    if split not in {"train", "test"}:
        raise ValueError("split doit valoir 'train' ou 'test'")

    image_key = "train_images" if split == "train" else "test_images"
    label_key = "train_labels" if split == "train" else "test_labels"

    images_path = cache_path / Path(MNIST_URLS[image_key]).name
    labels_path = cache_path / Path(MNIST_URLS[label_key]).name

    _download_file(MNIST_URLS[image_key], images_path)
    _download_file(MNIST_URLS[label_key], labels_path)

    images = _load_mnist_images(images_path)
    labels = _load_mnist_labels(labels_path)

    data: TrainData = []
    for image, label in zip(images, labels, strict=True):
        data.append(((image.reshape(1, image.shape[0], image.shape[1]),), (_one_hot(int(label)),)))

    return data
