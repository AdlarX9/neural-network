from __future__ import annotations
from core import (
    Network,
    Layer,
    Conv,
    FC,
    Biais,
    ConvBiais,
    ReLU,
    Flatten,
    ExitLoss,
    ProbaExit,
    Block,
    BN,
    Pool,
    Res,
    Embedding,
    Recurrent,
    LSTM,
    Decoder,
)
import os
import struct
from io import BufferedWriter, BufferedReader
from typing import Any
from pathlib import Path

layer_types = {
    "Layer": Layer,
    "Conv": Conv,
    "FC": FC,
    "Biais": Biais,
    "ConvBiais": ConvBiais,
    "ReLU": ReLU,
    "Flatten": Flatten,
    "ExitLoss": ExitLoss,
    "ProbaExit": ProbaExit,
    "Network": Network,
    "BN": BN,
    "Pool": Pool,
    "Res": Res,
    "Embedding": Embedding,
    "Recurrent": Recurrent,
    "LSTM": LSTM,
    "Decoder": Decoder,
}


class SaveHandler:
    _instance: SaveHandler | None = None

    def __new__(cls) -> SaveHandler:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self: SaveHandler) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

    def get_path(self: SaveHandler, name: str) -> str:
        return os.path.join("data", "models", name + ".network")

    def has(self: SaveHandler, name: str) -> bool:
        path = self.get_path(name)
        path = Path(path)
        return path.exists()

    def write_number(self: SaveHandler, f: BufferedWriter, val: float | int, type: str) -> None:
        f.write(struct.pack(type, val))

    def read_number(self: SaveHandler, f: BufferedReader, type: str) -> float | int:
        val = struct.unpack(type, f.read(4))[0]
        return val

    def write_list(self: SaveHandler, f: BufferedWriter, values: tuple | list, type: str) -> None:
        self.write_number(f, len(values), "i")
        f.write(struct.pack(f"{len(values)}{type}", *values))

    def read_list(self: SaveHandler, f: BufferedReader, type: str) -> list[Any]:
        length = struct.unpack("i", f.read(4))[0]
        values = list(struct.unpack(f"{length}{type}", f.read(4 * length)))
        return values

    def write_string(self: SaveHandler, f: BufferedWriter, string: str) -> None:
        self.write_number(f, len(string), "i")
        f.write(string.encode("utf-8"))

    def read_string(self: SaveHandler, f: BufferedReader) -> str:
        length = struct.unpack("i", f.read(4))[0]
        s = f.read(length).decode("utf-8")
        return s

    def write_string_list(self: SaveHandler, f: BufferedWriter, string_list: list[str]) -> None:
        self.write_number(f, len(string_list), "i")
        for string in string_list:
            self.write_string(f, string)

    def read_string_list(self: SaveHandler, f: BufferedReader) -> list[str]:
        length = self.read_number(f, "i")
        string_list = []
        for _ in range(int(length)):
            string = self.read_string(f)
            string_list.append(string)
        return string_list

    def write_layer(self: SaveHandler, f: BufferedWriter, layer: Layer) -> None:
        self.write_string(f, layer.__class__.__name__)
        int_list, float_list, string_list = layer.get_data()
        self.write_list(f, int_list, "i")
        self.write_list(f, float_list, "f")
        self.write_string_list(f, string_list)

    def read_layer(self: SaveHandler, f: BufferedReader) -> Layer:
        layer_type = self.read_string(f)
        int_list = self.read_list(f, "i")
        float_list = self.read_list(f, "f")
        string_list = self.read_string_list(f)
        new_layer: Layer = layer_types[layer_type]()
        if isinstance(new_layer, Block):
            new_layer.load_from_data(int_list, float_list, string_list, layer_types)
        else:
            new_layer.load_from_data(int_list, float_list, string_list)
        return new_layer

    def save(self: SaveHandler, layer: Layer, name: str) -> None:
        path = self.get_path(name)
        with open(path, "wb") as f:
            self.write_layer(f, layer)

    def load(self: SaveHandler, name: str) -> Layer:
        path = self.get_path(name)
        with open(path, "rb") as f:
            layer = self.read_layer(f)
        return layer

    def test(self: SaveHandler):
        name = "test"
        path = self.get_path(name)
        string = "hey moi c'est Alexis !"
        number = 189
        other_number = 25.3
        int_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        float_list = [0.1, 0.2, 0.3]
        with open(path, "wb") as f:
            self.write_string(f, string)
            self.write_number(f, number, "i")
            self.write_number(f, other_number, "f")
            self.write_list(f, int_list, "i")
            self.write_list(f, float_list, "f")
        with open(path, "rb") as f:
            decoded_string = self.read_string(f)
            print(string, "=>", decoded_string)
            decoded_number = self.read_number(f, "i")
            print(number, "=>", decoded_number)
            decoded_other_number = self.read_number(f, "f")
            print(other_number, "=>", decoded_other_number)
            decoded_int_list = self.read_list(f, "i")
            print(int_list, "=>", decoded_int_list)
            decoded_float_list = self.read_list(f, "f")
            print(float_list, "=>", decoded_float_list)
