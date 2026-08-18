from __future__ import annotations
import core
import os
import json
from pathlib import Path

layer_types = {
    "Layer": core.Layer,
    "Conv": core.Conv,
    "FC": core.FC,
    "Biais": core.Biais,
    "ReLU": core.ReLU,
    "ExitLoss": core.ExitLoss,
    "ProbaExit": core.ProbaExit,
    "Network": core.Network,
    "MLP": core.MLP,
    "CNN": core.CNN,
    "LSTMNetwork": core.LSTMNetwork,
    "BN": core.BN,
    "MaxPooling": core.MaxPooling,
    "Res": core.Res,
    "Embedding": core.Embedding,
    "Recurrent": core.Recurrent,
    "LSTM": core.LSTM,
    "OneHotMaker": core.OneHotMaker,
    "Adder": core.Adder,
    "Multiplier": core.Multiplier,
    "Activation": core.Activation,
    "GlobalAveragePooling": core.GlobalAveragePooling,
    "SiLU": core.SiLU,
    "SwiGLU": core.SwiGLU,
    "Linear": core.Linear,
    "LLaMA": core.LLaMA,
    "Block": core.Block,
    "RMSNorm": core.RMSNorm,
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
