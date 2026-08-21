from __future__ import annotations
import core
import os
import json
from pathlib import Path
import inspect
import struct
from io import BufferedWriter, BufferedReader
from typing import Any
import numpy as np

layer_types = {
    name: obj for name, obj in inspect.getmembers(core, inspect.isclass) if issubclass(obj, core.Layer)
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
        return os.path.join("data", "models", name)

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

    def save_parameters(self: SaveHandler, layer: core.Layer, path: str) -> None:
        params_data = layer.get_params_data()
        with open(path, "wb") as f:
            self.write_number(f, len(params_data), "i")
            for data in params_data:
                self.write_number(f, data["id"], "i")
                self.write_string_list(f, data["parameters"])
                for param_name in data["parameters"]:
                    self.write_list(f, data[param_name].shape, "i")
                    self.write_list(f, data[param_name].flatten().tolist(), "f")

    def load_parameters(self: SaveHandler, layer: core.Layer, path: str) -> None:
        with open(path, "rb") as f:
            len_params_data = self.read_number(f, "i")
            for _ in range(int(len_params_data)):
                layer_id = self.read_number(f, "i")
                parameters = self.read_string_list(f)
                parameterized = layer.get_layer_by_id(int(layer_id))
                if parameterized is None:
                    raise MemoryError
                for param_name in parameters:
                    shape = tuple(self.read_list(f, "i"))
                    param = self.read_list(f, "f")
                    param = np.array(param).reshape(shape)
                    setattr(parameterized, param_name, param)

    def save_json(self: SaveHandler, layer: core.Layer, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            data = layer.get_data()
            data["class"] = layer.__class__.__name__
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_json(self: SaveHandler, path: str) -> core.Layer:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            layer: core.Layer = layer_types[data["class"]]()
            if isinstance(layer, core.Block):
                layer.load_from_data(data, layer_types)
            else:
                layer.load_from_data(data)
        return layer

    def save(self: SaveHandler, layer: core.Layer, name: str) -> None:
        layer.set_id()
        base = self.get_path(name)
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "info.json")
        self.save_json(layer, path)
        path = os.path.join(base, "weights.bin")
        self.save_parameters(layer, path)

    def load(self: SaveHandler, name: str) -> core.Layer:
        base = self.get_path(name)
        path = os.path.join(base, "info.json")
        layer = self.load_json(path)
        path = os.path.join(base, "weights.bin")
        self.load_parameters(layer, path)
        return layer
