from __future__ import annotations

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import messagebox, scrolledtext

if TYPE_CHECKING:
    from core import GPT


class ChatWindow:
    def __init__(self, master: tk.Tk, gpt: GPT, temperature: float | None = None) -> None:
        self.master = master
        self.gpt = gpt
        self.temperature = temperature
        self.temperature_enabled = tk.BooleanVar(value=temperature is not None)
        initial_temperature = temperature if temperature is not None and temperature > 0 else 1.0
        self.temperature_value = tk.DoubleVar(value=initial_temperature)
        self.master.title("GPT Chat")
        self.master.minsize(820, 560)

        self.bg = "#f5f7fb"
        self.card = "#ffffff"
        self.text = "#111827"
        self.muted = "#6b7280"
        self.border = "#dbe3ee"
        self.accent = "#2563eb"
        self.accent_hover = "#1d4ed8"
        self.soft = "#eef4ff"

        self.master.configure(bg=self.bg)

        container = tk.Frame(self.master, bg=self.bg)
        container.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        title = tk.Label(
            container,
            text="Chat de génération de texte",
            fg=self.text,
            bg=self.bg,
            font=("Helvetica", 20, "bold"),
        )
        title.pack(anchor=tk.W)

        subtitle = tk.Label(
            container,
            text="Entre un prompt, choisis le nombre de mots à générer, puis lance la complétion.",
            fg=self.muted,
            bg=self.bg,
            font=("Helvetica", 11),
        )
        subtitle.pack(anchor=tk.W, pady=(4, 16))

        card = tk.Frame(
            container,
            bg=self.card,
            highlightbackground=self.border,
            highlightthickness=1,
            bd=0,
        )
        card.pack(fill=tk.BOTH, expand=True)

        self.history = scrolledtext.ScrolledText(
            card,
            wrap=tk.WORD,
            height=18,
            bg=self.card,
            fg=self.text,
            insertbackground=self.text,
            relief=tk.FLAT,
            font=("Helvetica", 12),
            padx=14,
            pady=14,
            borderwidth=0,
            highlightthickness=0,
        )
        self.history.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.history.configure(state=tk.DISABLED)

        controls = tk.Frame(container, bg=self.bg)
        controls.pack(fill=tk.X, pady=(16, 0))

        controls_card = tk.Frame(
            controls,
            bg=self.card,
            highlightbackground=self.border,
            highlightthickness=1,
            bd=0,
        )
        controls_card.pack(fill=tk.X)

        inner = tk.Frame(controls_card, bg=self.card)
        inner.pack(fill=tk.X, padx=16, pady=16)

        prompt_label = tk.Label(
            inner,
            text="Prompt",
            fg=self.text,
            bg=self.card,
            font=("Helvetica", 11, "bold"),
        )
        prompt_label.grid(row=0, column=0, sticky=tk.W)

        self.prompt_entry = tk.Entry(
            inner,
            bg="#f8fafc",
            fg=self.text,
            insertbackground=self.text,
            relief=tk.FLAT,
            font=("Helvetica", 12),
            highlightthickness=1,
            highlightbackground=self.border,
            highlightcolor=self.accent,
            bd=0,
        )
        self.prompt_entry.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 12), ipady=7)
        self.prompt_entry.insert(0, "je ne suis pas")

        count_label = tk.Label(
            inner,
            text="Mots à générer",
            fg=self.text,
            bg=self.card,
            font=("Helvetica", 11, "bold"),
        )
        count_label.grid(row=2, column=0, sticky=tk.W)

        self.word_count = tk.Spinbox(
            inner,
            from_=1,
            to=200,
            width=8,
            bg="#f8fafc",
            fg=self.text,
            insertbackground=self.text,
            buttonbackground="#f8fafc",
            relief=tk.FLAT,
            font=("Helvetica", 12),
            highlightthickness=1,
            highlightbackground=self.border,
            highlightcolor=self.accent,
            bd=0,
        )
        self.word_count.grid(row=3, column=0, sticky=tk.W, pady=(6, 0), ipady=5)
        self.word_count.delete(0, tk.END)
        self.word_count.insert(0, "80")

        generate_button = tk.Button(
            inner,
            text="Générer",
            command=self.generate,
            bg=self.accent,
            fg="white",
            activebackground=self.accent_hover,
            activeforeground="white",
            relief=tk.FLAT,
            font=("Helvetica", 11, "bold"),
            padx=18,
            pady=7,
        )
        generate_button.grid(row=3, column=1, sticky=tk.W, padx=(14, 0), pady=(6, 0))

        clear_button = tk.Button(
            inner,
            text="Effacer l'historique",
            command=self.clear_history,
            bg=self.soft,
            fg=self.text,
            activebackground="#dbeafe",
            activeforeground=self.text,
            relief=tk.FLAT,
            font=("Helvetica", 11),
            padx=18,
            pady=7,
        )
        clear_button.grid(row=3, column=2, sticky=tk.W, padx=(10, 0), pady=(6, 0))

        temperature_check = tk.Checkbutton(
            inner,
            text="Activer la température",
            variable=self.temperature_enabled,
            command=self._toggle_temperature_slider,
            bg=self.card,
            fg=self.text,
            activebackground=self.card,
            activeforeground=self.text,
            selectcolor=self.card,
            font=("Helvetica", 11),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
        )
        temperature_check.grid(row=4, column=0, sticky=tk.W, pady=(14, 0))

        self.temperature_scale = tk.Scale(
            inner,
            from_=0.01,
            to=2.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.temperature_value,
            bg=self.card,
            fg=self.text,
            troughcolor="#e5e7eb",
            highlightthickness=0,
            relief=tk.FLAT,
            length=260,
            label="Température",
            font=("Helvetica", 10),
        )
        self.temperature_scale.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        self._toggle_temperature_slider()

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=0)
        inner.columnconfigure(2, weight=0)

        footer = tk.Label(
            container,
            text="Entrée pour générer, Échap pour fermer.",
            fg=self.muted,
            bg=self.bg,
            font=("Helvetica", 10),
        )
        footer.pack(anchor=tk.W, pady=(10, 0))

        self.prompt_entry.bind("<Return>", lambda _event: self.generate())
        self.master.bind("<Escape>", lambda _event: self.master.destroy())

        self._append_message("Système", "Prêt. Entre un texte et lance la génération.")
        self.prompt_entry.focus_set()

    def _toggle_temperature_slider(self) -> None:
        state = tk.NORMAL if self.temperature_enabled.get() else tk.DISABLED
        self.temperature_scale.configure(state=state)

    def _append_message(self, speaker: str, message: str) -> None:
        self.history.configure(state=tk.NORMAL)
        self.history.insert(tk.END, f"{speaker} : {message}\n\n")
        self.history.see(tk.END)
        self.history.configure(state=tk.DISABLED)

    def clear_history(self) -> None:
        self.history.configure(state=tk.NORMAL)
        self.history.delete("1.0", tk.END)
        self.history.configure(state=tk.DISABLED)

    def generate(self) -> None:
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Prompt vide", "Entre au moins un texte de départ.")
            return

        try:
            word_count = int(self.word_count.get())
        except ValueError:
            messagebox.showerror("Valeur invalide", "Le nombre de mots doit être un entier.")
            return

        if word_count < 1:
            messagebox.showerror("Valeur invalide", "Le nombre de mots doit être supérieur à 0.")
            return

        selected_temperature: float | None = None
        if self.temperature_enabled.get():
            selected_temperature = float(self.temperature_value.get())
            if selected_temperature <= 0:
                messagebox.showerror("Valeur invalide", "La température doit être strictement positive.")
                return

        try:
            completed_sentence = self.gpt.generate(prompt, word_count, selected_temperature)
        except Exception as exc:
            messagebox.showerror("Erreur de génération", str(exc))
            return

        self._append_message("Vous", prompt)
        self._append_message("GPT", completed_sentence)


def chat(gpt: GPT, temperature: float | None = None) -> None:
    root = tk.Tk()
    ChatWindow(root, gpt, temperature)
    root.mainloop()
