from __future__ import annotations
import core
import os
import json
from pathlib import Path
import inspect

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
        return os.path.join("data", "models", name + ".json")

    def has(self: SaveHandler, name: str) -> bool:
        path = self.get_path(name)
        path = Path(path)
        return path.exists()

    def save(self: SaveHandler, layer: core.Layer, name: str) -> None:
        path = self.get_path(name)
        with open(path, "w", encoding="utf-8") as f:
            data = layer.get_data()
            data["class"] = layer.__class__.__name__
            json.dump(data, f, ensure_ascii=False)

    def load(self: SaveHandler, name: str) -> core.Layer:
        path = self.get_path(name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            layer: core.Layer = layer_types[data["class"]]()
            if isinstance(layer, core.Block):
                layer.load_from_data(data, layer_types)
            else:
                layer.load_from_data(data)
        return layer
