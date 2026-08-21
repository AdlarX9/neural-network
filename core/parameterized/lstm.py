from __future__ import annotations
from typing import Any
from ..basics.layer import Layer
import numpy as np
from numpy.typing import NDArray
from ..utils.functions import sigmoid
from ..utils.typing import ShapeFlow, Tensor, Receive1, SaveData, ParamGrad


class LSTM(Layer):
    def __init__(self: LSTM, receive: Receive1 = (0,)) -> None:
        super().__init__(receive)
        self.data: list[dict[str, Tensor]] = []
        self.n = 0
        self.h = np.array([[]])
        self.c = np.array([[]])

        # Input weights
        self.Wf = np.array([[]])
        self.Wi = np.array([[]])
        self.Wo = np.array([[]])
        self.Wc = np.array([[]])

        # ST weights
        self.Uf = np.array([[]])
        self.Ui = np.array([[]])
        self.Uo = np.array([[]])
        self.Uc = np.array([[]])

        # Biaises
        self.bf = np.array([[]])
        self.bi = np.array([[]])
        self.bo = np.array([[]])
        self.bc = np.array([[]])

        self.parameters = ["Wf", "Wi", "Wo", "Wc", "Uf", "Ui", "Uo", "Uc", "bf", "bi", "bo", "bc"]

        self.gradient_h: Tensor | None = None
        self.gradient_c: Tensor | None = None
        self.param_grad: ParamGrad | None = None

    def reset_data(self: LSTM) -> None:
        self.h = np.zeros_like(self.h)
        self.c = np.zeros_like(self.c)
        self.data = [
            {
                "f": np.zeros_like(self.h),
                "o": np.zeros_like(self.h),
                "i": np.zeros_like(self.h),
                "c_prime": np.zeros_like(self.h),
                "h": np.zeros_like(self.h),
                "c": np.zeros_like(self.h),
                "x": np.zeros_like(self.h),
            }
        ]
        self.gradient_h = None
        self.gradient_c = None

    def set_input_shape(self: LSTM, input_shape: ShapeFlow) -> ShapeFlow:
        n, p = input_shape[0]
        self.n = n
        if p != 1:
            raise ValueError
        super().set_input_shape(input_shape)
        self.h = np.zeros((self.n, p))
        self.c = np.zeros((self.n, p))

        # Input weights
        self.Wf = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))
        self.Wi = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))
        self.Wo = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))
        self.Wc = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))

        # ST weights
        self.Uf = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))
        self.Ui = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))
        self.Uo = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))
        self.Uc = np.random.normal(0, np.sqrt(2 / (self.n * n)), size=(self.n, n))

        # Biaises
        self.bf = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))
        self.bi = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))
        self.bo = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))
        self.bc = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))

        self.reset_data()
        return self.output_shape

    def feed_forward(self: LSTM, entry: Tensor) -> Tensor:
        forget = sigmoid(self.Wf @ entry + self.Uf @ self.h + self.bf)
        input = sigmoid(self.Wi @ entry + self.Ui @ self.h + self.bi)
        output = sigmoid(self.Wo @ entry + self.Uo @ self.h + self.bo)
        c_prime = np.tanh(self.Wc @ entry + self.Uc @ self.h + self.bc)
        self.c = forget * self.c + input * c_prime
        self.h = output * np.tanh(self.c)

        self.data.append(
            {
                "f": forget,
                "o": output,
                "i": input,
                "c_prime": c_prime,
                "h": self.h,
                "c": self.c,
                "x": entry,
            }
        )

        return self.h

    def descend_gradient(self: LSTM, gradient: Tensor) -> Tensor:
        if len(self.data) < 2:
            raise MemoryError
        if self.gradient_h is None:
            self.gradient_h = gradient

        # Extract memoized value
        f = self.data[-1]["f"]
        o = self.data[-1]["o"]
        i = self.data[-1]["i"]
        c_prime = self.data[-1]["c_prime"]
        c = self.data[-1]["c"]
        x = self.data[-1]["x"]
        h_before = self.data[-2]["h"]
        c_before = self.data[-2]["c"]
        self.data.pop()

        # Compute gradients
        tanh_c = np.tanh(c)
        do = self.gradient_h * tanh_c
        dc = self.gradient_h * o * (1 - tanh_c**2)
        if self.gradient_c is not None:
            dc += self.gradient_c
        df = c_before * dc
        di = dc * c_prime
        dc_prime = dc * i

        # Compute useful variables
        derivate_f = df * f * (1 - f)
        derivate_i = di * i * (1 - i)
        derivate_o = do * o * (1 - o)
        derivate_c_prime = dc_prime * (1 - c_prime**2)

        # Compute new gradients
        self.gradient_h = (
            self.Uf.T @ derivate_f
            + self.Ui.T @ derivate_i
            + self.Uo.T @ derivate_o
            + self.Uc.T @ derivate_c_prime
        )
        self.gradient_c = dc * f
        new_gradient = (
            self.Wf.T @ derivate_f
            + self.Wi.T @ derivate_i
            + self.Wo.T @ derivate_o
            + self.Wc.T @ derivate_c_prime
        )

        # Learn weights
        self.param_grad = {
            "Wf": derivate_f @ x.T,
            "Wi": derivate_i @ x.T,
            "Wo": derivate_o @ x.T,
            "Wc": derivate_c_prime @ x.T,
            "Uf": derivate_f @ h_before.T,
            "Ui": derivate_i @ h_before.T,
            "Uo": derivate_o @ h_before.T,
            "Uc": derivate_c_prime @ h_before.T,
            "bf": derivate_f,
            "bi": derivate_i,
            "bo": derivate_o,
            "bc": derivate_c_prime,
        }

        return new_gradient

    def params_gradient(self: LSTM, gradient) -> ParamGrad:
        if self.param_grad is None:
            raise MemoryError
        return self.param_grad

    def get_data(self: LSTM) -> SaveData:
        data = super().get_data()
        data["n"] = self.n
        return data

    def load_from_data(self: LSTM, data: SaveData) -> None:
        super().load_from_data(data)
        self.n = data["n"]
        self.h = np.zeros(self.input_shape[0])
        self.c = np.zeros(self.input_shape[0])
