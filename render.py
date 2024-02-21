from functools import lru_cache
import tkinter as tk
from PIL import Image, ImageTk

from utils import ignore_unhashable


class ImageWindow:
    def __init__(self, master: tk.Tk, title: str = "Rendered Image"):
        self.master = master
        self.master.title(title)
        self.label = tk.Label(self.master)
        self.label.pack()

    @ignore_unhashable
    @lru_cache(maxsize=10)
    def update_image(self, image: list[float]) -> None:
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.label.configure(image=photo)
        self.label.image = photo
