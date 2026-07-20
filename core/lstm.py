from __future__ import annotations
from .layer import Layer
import numpy as np
from numpy.typing import NDArray
from .functions import sigmoid


class LSTM(Layer):
    def __init__(self: LSTM) -> None:
        super().__init__()
        self.data: list[dict[str, NDArray[np.float64]]] = []
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

        self.gradient_h: NDArray[np.float64] | None = None
        self.gradient_c: NDArray[np.float64] | None = None

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

    def set_input_shape(self: LSTM, input_shape: tuple) -> tuple:
        n, p = input_shape
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

    def feed_forward(self: LSTM, entry: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def descend_gradient(self: LSTM, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
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
            self.Uf @ derivate_f + self.Ui @ derivate_i + self.Uo @ derivate_o + self.Uc @ derivate_c_prime
        )
        self.gradient_c = dc * f
        new_gradient = (
            self.Wf @ derivate_f + self.Wi @ derivate_i + self.Wo @ derivate_o + self.Wc @ derivate_c_prime
        )

        # Learn weights
        self.Wf -= self.lr * derivate_f @ x.T
        self.Wi -= self.lr * derivate_i @ x.T
        self.Wo -= self.lr * derivate_o @ x.T
        self.Wc -= self.lr * derivate_c_prime @ x.T
        self.Uf -= self.lr * derivate_f @ h_before.T
        self.Ui -= self.lr * derivate_i @ h_before.T
        self.Uo -= self.lr * derivate_o @ h_before.T
        self.Uc -= self.lr * derivate_c_prime @ h_before.T
        self.bf -= self.lr * derivate_f
        self.bi -= self.lr * derivate_i
        self.bo -= self.lr * derivate_o
        self.bc -= self.lr * derivate_c_prime

        return new_gradient

    def get_data(self: LSTM) -> tuple[list[int], list[float], list[str]]:
        int_list, float_list, str_list = super().get_data()
        float_list += self.Wf.flatten().tolist()
        float_list += self.Wi.flatten().tolist()
        float_list += self.Wo.flatten().tolist()
        float_list += self.Wc.flatten().tolist()
        float_list += self.Uf.flatten().tolist()
        float_list += self.Ui.flatten().tolist()
        float_list += self.Uo.flatten().tolist()
        float_list += self.Uc.flatten().tolist()
        float_list += self.bf.flatten().tolist()
        float_list += self.bi.flatten().tolist()
        float_list += self.bo.flatten().tolist()
        float_list += self.bc.flatten().tolist()
        return int_list, float_list, str_list

    def load_from_data(
        self: LSTM, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.input_shape = tuple(int_list[:2])
        self.n = self.input_shape[0]
        del int_list[:2]
        self.lr = float_list[0]
        float_list.pop(0)
        self.h = np.zeros((self.n, 1))
        self.c = np.zeros((self.n, 1))

        self.Wf = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]
        self.Wi = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]
        self.Wo = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]
        self.Wc = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]

        self.Uf = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]
        self.Ui = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]
        self.Uo = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]
        self.Uc = np.array(float_list[: self.n**2]).reshape((self.n, self.n))
        del float_list[: self.n**2]

        self.bf = np.array(float_list[: self.n]).reshape((self.n, 1))
        del float_list[: self.n]
        self.bi = np.array(float_list[: self.n]).reshape((self.n, 1))
        del float_list[: self.n]
        self.bo = np.array(float_list[: self.n]).reshape((self.n, 1))
        del float_list[: self.n]
        self.bc = np.array(float_list[: self.n]).reshape((self.n, 1))
        del float_list[: self.n]
