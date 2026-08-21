import numpy as np
from numpy.typing import NDArray
from typing import Any, TypeAlias

Shape: TypeAlias = tuple[int, ...]
ShapeFlow: TypeAlias = tuple[Shape, ...]

Tensor: TypeAlias = NDArray[np.float64]
TensorFlow: TypeAlias = tuple[Tensor, ...]
ParamGrad: TypeAlias = dict[str, Tensor]

Tokens: TypeAlias = list[int]

SaveData: TypeAlias = dict[str, Any]

Batch: TypeAlias = tuple[TensorFlow, TensorFlow]
TrainData: TypeAlias = list[Batch]

Receive: TypeAlias = tuple[int, ...]
Receive1: TypeAlias = tuple[int]
Receive2: TypeAlias = tuple[int, int]
