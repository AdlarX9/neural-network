from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Callable
import tkinter as tk

if TYPE_CHECKING:
    from core import Network

WIDTH, HEIGHT = 2000, 800
PLOT_SIZE = 680
GRID_SIZE = 80


def regression(
    network: Network,
    teacher: Callable,
) -> None:
    network_dimensions = network.get_dimensions()
    if network_dimensions[0][0][0] != 2 or network_dimensions[1][0][0] != 1:
        return

    root = tk.Tk()
    root.title("2D Regression")
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
    canvas.pack()

    values = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    correction = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = (i + 0.5) / GRID_SIZE
            y = (j + 0.5) / GRID_SIZE
            inp = np.array([x, y]).reshape(-1, 1)
            prediction = network((inp,))[0]
            correction[j, i] = teacher(x, y)
            val = prediction[0, 0]
            values[j, i] = val

    vmin1 = float(np.nanmin(values))
    vmin2 = float(np.nanmin(correction))
    vmax1 = float(np.nanmax(values))
    vmax2 = float(np.nanmax(correction))
    vmin = min(vmin1, vmin2)
    vmax = max(vmax1, vmax2)

    cell_size = PLOT_SIZE / GRID_SIZE
    margin = 100

    plot_left = (WIDTH + margin) / 2
    plot_top = (HEIGHT - PLOT_SIZE) / 2
    plot_right = plot_left + PLOT_SIZE
    plot_bottom = plot_top + PLOT_SIZE

    correction_right = (WIDTH - margin) / 2
    correction_left = correction_right - PLOT_SIZE

    def value_to_rgb(v: float) -> tuple[int, int, int]:
        if abs(vmin - vmax) < 1e-3:
            return (255, 255, 255)
        t = (v - vmin) / (vmax - vmin)
        r = int(255 * (1 - t))
        g = int(255 * t)
        return r, g, 0

    canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="white", width=2)
    canvas.create_rectangle(
        correction_left,
        plot_top,
        correction_right,
        plot_bottom,
        outline="white",
        width=2,
    )

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            v = values[j, i]
            v_ = correction[j, i]
            r, g, b = value_to_rgb(v)
            r_, g_, b_ = value_to_rgb(v_)
            color = f"#{r:02x}{g:02x}{b:02x}"
            color_ = f"#{r_:02x}{g_:02x}{b_:02x}"
            x1 = plot_left + i * cell_size
            x1_ = correction_left + i * cell_size
            y1 = plot_bottom - j * cell_size
            x2 = x1 + cell_size
            x2_ = x1_ + cell_size
            y2 = y1 - cell_size
            canvas.create_rectangle(x1, y1, x2, y2, outline="", fill=color)
            canvas.create_rectangle(x1_, y1, x2_, y2, outline="", fill=color_)

    canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill="white", width=2)
    canvas.create_line(plot_left, plot_top, plot_left, plot_bottom, fill="white", width=2)

    font_size = max(10, int(PLOT_SIZE / 40))
    canvas.create_text(
        plot_left - 12,
        plot_bottom + 12,
        text="0",
        fill="white",
        font=("Helvetica", font_size),
    )
    canvas.create_text(
        plot_right + 12,
        plot_bottom + 12,
        text="1",
        fill="white",
        font=("Helvetica", font_size),
    )
    canvas.create_text(
        plot_left - 12,
        plot_top + 12,
        text="1",
        fill="white",
        font=("Helvetica", font_size),
    )
    canvas.create_text(
        correction_left - 12,
        plot_bottom + 12,
        text="0",
        fill="white",
        font=("Helvetica", font_size),
    )
    canvas.create_text(
        correction_right + 12,
        plot_bottom + 12,
        text="1",
        fill="white",
        font=("Helvetica", font_size),
    )
    canvas.create_text(
        correction_left - 12,
        plot_top + 12,
        text="1",
        fill="white",
        font=("Helvetica", font_size),
    )

    canvas.create_text(
        plot_left,
        plot_top - 18,
        text=f"min = {vmin1:.2f}",
        fill="#ff4444",
        anchor="w",
        font=("Helvetica", font_size, "bold"),
    )
    canvas.create_text(
        plot_right,
        plot_top - 18,
        text=f"max = {vmax1:.2f}",
        fill="#00ff00",
        anchor="e",
        font=("Helvetica", font_size, "bold"),
    )
    canvas.create_text(
        correction_left,
        plot_top - 18,
        text=f"min = {vmin2:.2f}",
        fill="#ff4444",
        anchor="w",
        font=("Helvetica", font_size, "bold"),
    )
    canvas.create_text(
        correction_right,
        plot_top - 18,
        text=f"max = {vmax2:.2f}",
        fill="#00ff00",
        anchor="e",
        font=("Helvetica", font_size, "bold"),
    )

    root.mainloop()
