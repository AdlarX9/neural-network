import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


class DiffusionWindow:
    def __init__(self, master: tk.Tk, diffusion, mode: str) -> None:
        self.master = master
        self.diffusion = diffusion
        self.mode = mode

        self.master.title("diffusion")
        self.master.geometry("800x700")
        self.master.minsize(500, 500)

        self.prompt_frame = tk.Frame(master)
        self.prompt_frame.pack(fill="x", padx=15, pady=15)

        self.prompt = tk.Entry(self.prompt_frame, font=("Arial", 16))
        self.prompt.pack(side="left", fill="x", expand=True)
        self.prompt.bind("<Return>", self.generate)

        self.button = tk.Button(
            self.prompt_frame,
            text="Générer",
            font=("Arial", 14),
            command=self.generate,
        )
        self.button.pack(side="left", padx=(10, 0))

        self.status = tk.Label(
            master,
            text="Entrez un prompt.",
            font=("Arial", 12),
        )
        self.status.pack(pady=(0, 10))

        self.image_label = tk.Label(master)
        self.image_label.pack(
            expand=True,
            fill="both",
            padx=15,
            pady=15,
        )

        self.photo = None

    def generate(self, event=None) -> None:
        text = self.prompt.get().strip()

        if not text:
            return

        self.button.config(state="disabled")
        self.status.config(text="Génération en cours...")
        self.master.update_idletasks()

        try:
            generate_method = getattr(self.diffusion, 'generate_' + self.mode)
            image = generate_method(text)

            if not isinstance(image, np.ndarray):
                raise TypeError("diffusion.generate() doit retourner un NDArray NumPy.")

            if image.ndim != 3:
                raise ValueError(f"L'image doit être un tenseur 3D (C, H, W), reçu {image.shape}.")

            channels, height, width = image.shape

            if channels not in (1, 3):
                raise ValueError(f"Le tenseur doit avoir 1 ou 3 canaux, reçu {channels}.")

            # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))

            # Conversion vers [0, 255].
            # Si le modèle produit déjà des valeurs dans [0, 1],
            # cette conversion est directe. Pour des valeurs [-1, 1],
            # on les remappe automatiquement.
            image = image.astype(np.float64)

            if np.min(image) < 0:
                image = (image + 1.0) / 2.0

            image = np.clip(image, 0.0, 1.0)
            image = (image * 255).astype(np.uint8)

            if channels == 1:
                image = image[:, :, 0]
                pil_image = Image.fromarray(image, mode="L")
            else:
                pil_image = Image.fromarray(image, mode="RGB")

            # Taille d'affichage constante tout en conservant le ratio.
            max_width = 650
            max_height = 550

            scale = min(
                max_width / width,
                max_height / height,
            )

            display_width = max(1, int(width * scale))
            display_height = max(1, int(height * scale))

            pil_image = pil_image.resize(
                (display_width, display_height),
                Image.Resampling.NEAREST,
            )

            self.photo = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=self.photo)

            self.status.config(text=f"{width} × {height} — {channels} canal{'aux' if channels > 1 else ''}")

        except Exception as error:
            self.status.config(text="Erreur")
            messagebox.showerror(
                "Erreur pendant la génération",
                str(error),
            )

        finally:
            self.button.config(state="normal")


def image_generator(diffusion, mode: str) -> None:
    root = tk.Tk()
    DiffusionWindow(root, diffusion, mode)
    root.mainloop()
