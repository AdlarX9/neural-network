from __future__ import annotations
import tkinter as tk
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core import Network

GRID_SIZE: Tuple[int, int] = (28, 28)
PIXEL_SIZE = 30
PRED_BAR_WIDTH = 240


class NumbersViewer:
    def __init__(self, master: tk.Tk, network: Network, grid_size=GRID_SIZE, pixel_size=PIXEL_SIZE):
        self.master = master
        self.network = network
        self.grid_w, self.grid_h = grid_size[1], grid_size[0]
        self.pixel_size = pixel_size

        self.data = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

        self._dragging = False
        self._pending_update = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.master.title("CNN Digit Drawer")

        container = tk.Frame(self.master)
        container.pack(fill=tk.BOTH, expand=True)

        # Left: drawing canvas
        left = tk.Frame(container)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        w = self.grid_w * self.pixel_size
        h = self.grid_h * self.pixel_size
        self.canvas = tk.Canvas(
            left, width=w, height=h, bg="black", highlightthickness=1, highlightbackground="gray"
        )
        self.canvas.pack()

        # Create rectangles
        self.rects = []
        for r in range(self.grid_h):
            row = []
            for c in range(self.grid_w):
                x0 = c * self.pixel_size
                y0 = r * self.pixel_size
                rect = self.canvas.create_rectangle(
                    x0, y0, x0 + self.pixel_size, y0 + self.pixel_size, fill="#000000", outline="#222"
                )
                row.append(rect)
            self.rects.append(row)

        self.canvas.bind("<B1-Motion>", self._on_paint)
        self.canvas.bind("<Button-1>", self._on_paint)
        self.canvas.bind("<B3-Motion>", self._on_erase)
        self.canvas.bind("<Button-3>", self._on_erase)

        # Controls under canvas
        btn_frame = tk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(6, 0))
        clear_btn = tk.Button(btn_frame, text="Réinitialiser", command=self.clear)
        clear_btn.pack(side=tk.LEFT)
        hint = tk.Label(btn_frame, text="Clic gauche: dessiner — Clic droit: effacer")
        hint.pack(side=tk.LEFT, padx=8)

        # Right: probabilities
        right = tk.Frame(container)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        self.pred_label = tk.Label(right, text="Prédiction : -", font=("Helvetica", 18))
        self.pred_label.pack(anchor=tk.NW)

        self.bar_items = []
        for d in range(10):
            row = tk.Frame(right)
            row.pack(fill=tk.X, pady=4)
            lbl = tk.Label(row, text=f"{d}", width=2)
            lbl.pack(side=tk.LEFT)
            bar = tk.Canvas(row, width=PRED_BAR_WIDTH, height=20, bg="#333", highlightthickness=0)
            bar.pack(side=tk.LEFT)
            bar_id = bar.create_rectangle(0, 0, 0, 20, fill="#4caf50")
            pct = tk.Label(row, text="0.00%", width=7)
            pct.pack(side=tk.LEFT, padx=6)
            self.bar_items.append((bar, bar_id, pct))

        # Status / errors
        self.status = tk.Label(self.master, text="", anchor=tk.W)
        self.status.pack(fill=tk.X)

        # Keyboard
        self.master.bind("c", lambda e: self.clear())

        # Initial prediction
        self._schedule_update()

    def _on_paint(self, event) -> None:
        x, y = event.x, event.y
        c = int(x // self.pixel_size)
        r = int(y // self.pixel_size)
        self._set_cell(r, c, 1.0)
        self._paint_neighbors(r, c, 1.0)
        self._schedule_update()

    def _on_erase(self, event) -> None:
        x, y = event.x, event.y
        c = int(x // self.pixel_size)
        r = int(y // self.pixel_size)
        self._set_cell(r, c, 0.0)
        self._paint_neighbors(r, c, 0.0)
        self._schedule_update()

    def _paint_neighbors(self, r: int, c: int, value: float) -> None:
        # simple 3x3 brush for smoother strokes
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                self._set_cell(r + dr, c + dc, value)

    def _set_cell(self, r: int, c: int, value: float) -> None:
        if r < 0 or c < 0 or r >= self.grid_h or c >= self.grid_w:
            return
        value = float(max(0.0, min(1.0, value)))
        if self.data[r, c] == value:
            return
        self.data[r, c] = value
        gray = int(value * 255)
        hexc = f"#{gray:02x}{gray:02x}{gray:02x}"
        self.canvas.itemconfig(self.rects[r][c], fill=hexc)

    def clear(self) -> None:
        self.data.fill(0.0)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                self.canvas.itemconfig(self.rects[r][c], fill="#000000")
        self._schedule_update()

    def _schedule_update(self, delay_ms: int = 50) -> None:
        if not self._pending_update:
            self._pending_update = True
            self.master.after(delay_ms, self._do_update)

    def _do_update(self) -> None:
        self._pending_update = False
        try:
            inp = self.data.reshape((1, self.grid_h, self.grid_w)).astype(np.float64)
            probs = self.network(inp)  # type: ignore
            probs = np.array(probs).reshape(-1)
            # ensure length 10
            if probs.size >= 10:
                probs = probs[:10]
            else:
                probs = np.pad(probs, (0, 10 - probs.size), constant_values=0.0)

            pred = int(np.argmax(probs))
            self.pred_label.config(text=f"Prédiction : {pred}")

            for i, (bar, bar_id, pct) in enumerate(self.bar_items):
                w = int(probs[i] * PRED_BAR_WIDTH)
                bar.coords(bar_id, 0, 0, w, 20)
                pct.config(text=f"{probs[i]*100:5.2f}%")

            self.status.config(text="")
        except Exception as exc:
            self.status.config(text=f"Erreur: {exc}")


def view_numbers(network: Network) -> None:
    root = tk.Tk()
    NumbersViewer(root, network)
    root.mainloop()
