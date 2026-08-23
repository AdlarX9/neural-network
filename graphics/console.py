from __future__ import annotations
import shutil
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core import Trainer


class ConsoleVisualization:
    def __init__(self: ConsoleVisualization, trainer: Trainer | None = None, stream=None) -> None:
        if trainer is not None:
            self.total_batches = max(1, trainer.total_batches)
            self.total_items = max(1, trainer.total_items)
            self.models = [layer.__class__.__name__ for layer in trainer.layers]
            self.parameters = [layer.get_parameters() for layer in trainer.layers]
        else:
            self.total_batches = 1
            self.total_items = 1
            self.models = []
            self.parameters = []
        self.stream = stream if stream is not None else sys.stdout
        self.started_at = time.perf_counter()
        self.batch_index = 0
        self.item_index = 0
        self.loss = 0.0
        self.accuracy = 0.0
        self._cursor_hidden = False
        self.max_fps = 20
        self.timestamp_last_render = 0.0
        self.title = "Network Training Dashboard"
        self._hide_cursor()

    def _hide_cursor(self) -> None:
        if not self._cursor_hidden:
            self.stream.write("\033[?25l")
            self.stream.flush()
            self._cursor_hidden = True

    def _show_cursor(self) -> None:
        if self._cursor_hidden:
            self.stream.write("\033[?25h")
            self.stream.flush()
            self._cursor_hidden = False

    def close(self) -> None:
        self._show_cursor()
        self.stream.write("\n")
        self.stream.flush()

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _progress_bar(self, ratio: float, width: int) -> str:
        ratio = min(1.0, max(0.0, ratio))
        filled = int(round(ratio * width))
        if filled >= width:
            return "#" * width
        return "#" * filled + ">" + "-" * max(0, width - filled - 1)

    def update(self, batch_index: int, item_index: int, loss: float, accuracy: float) -> None:
        self.batch_index = batch_index
        self.item_index = item_index
        self.loss = loss
        self.accuracy = accuracy
        if time.perf_counter() - self.timestamp_last_render >= 1 / self.max_fps:
            self.timestamp_last_render = time.perf_counter()
            self.render()

    def render(self, final: bool = False) -> None:
        elapsed = time.perf_counter() - self.started_at
        completed_items = max(0, (self.batch_index - 1) * self.total_items + self.item_index)
        total_items = self.total_batches * self.total_items
        progress = min(1.0, completed_items / total_items) if total_items else 1.0

        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            remaining = 0.0

        term_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        bar_width = max(20, min(50, term_width - 30))
        progress_bar = self._progress_bar(progress, bar_width)

        batch_progress = self.item_index / self.total_items if self.total_items else 1.0
        batch_completed = self.batch_index - 1 + batch_progress
        batch_remaining = self.total_batches - batch_completed
        models = ", ".join(
            [
                f'{self.models[i]} ({f"{self.parameters[i]:,}".replace(",", " ")})'
                for i in range(len(self.models))
            ]
        )

        lines = [
            "╭" + "─" * max(28, term_width - 2) + "╮",
            "│ " + self.title + " " * max(0, term_width - 3 - len(self.title)) + "│",
            "├" + "─" * max(28, term_width - 2) + "┤",
            f"│ Modèle               : {models}"
            + " " * max(0, term_width - 26 - len(models))
            + "│",
            f"│ Temps écoulé         : {self._format_duration(elapsed)}"
            + " " * max(0, term_width - 26 - len(self._format_duration(elapsed)))
            + "│",
            f"│ Temps restant estimé : {self._format_duration(remaining)}"
            + " " * max(0, term_width - 26 - len(self._format_duration(remaining)))
            + "│",
            f"│ Batch actuel         : {self.batch_index}/{self.total_batches} (restant {batch_remaining:.2f})"
            + " "
            * max(
                0,
                term_width
                - 26
                - len(f"{self.batch_index}/{self.total_batches} (restant {batch_remaining:.2f})"),
            )
            + "│",
            f"│ Données batch        : {self.item_index}/{self.total_items}"
            + " " * max(0, term_width - 26 - len(f"{self.item_index}/{self.total_items}"))
            + "│",
            f"│ Loss                 : {self.loss:.6f}"
            + " " * max(0, term_width - 26 - len(f"{self.loss:.6f}"))
            + "│",
            f"│ Accuracy             : {self.accuracy * 100:6.2f}%"
            + " " * max(0, term_width - 26 - len(f"{self.accuracy * 100:6.2f}%"))
            + "│",
            "├" + "─" * max(28, term_width - 2) + "┤",
            "│ Progression totale" + " " * max(0, term_width - 21) + "│",
            f"│ [{progress_bar}] {progress * 100:6.2f}%" + " " * max(0, term_width - 13 - bar_width) + "│",
            "╰" + "─" * max(28, term_width - 2) + "╯",
        ]

        if final:
            self.stream.write("\033[H")
        else:
            self.stream.write("\033[2J\033[H")
        self.stream.write("\n".join(lines))
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
