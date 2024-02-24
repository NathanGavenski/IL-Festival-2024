from functools import lru_cache
import tkinter as tk
from PIL import Image, ImageTk

from utils import ignore_unhashable


class ImageWindow:
    def __init__(
        self,
        master: tk.Tk,
        title: str = "Rendered Image",
        size: float = 3.5
    ):
        self.master = master
        self.master.title(title)
        self.label = tk.Label(self.master)
        self.label.pack()
        self.ratio = size

    @ignore_unhashable
    @lru_cache(maxsize=10)
    def update_image(self, image: list[float]) -> None:
        width, height, *_ = image.shape
        image = Image.fromarray(image)
        image = image.resize(
            (int(width * self.ratio), int(height * self.ratio)),
            Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(image)
        self.label.configure(image=photo)
        self.label.image = photo
