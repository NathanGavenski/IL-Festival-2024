"""Module for the window to display rendered images."""
from functools import lru_cache
import tkinter as tk

from PIL import Image, ImageTk

from utils import ignore_unhashable


class ImageWindow:
    """A window to display rendered images.

    Parameters:
        master: The parent window.
        title: The title of the window.
        size: The size of the rendered image.
    """
    def __init__(
        self,
        master: tk.Tk,
        title: str = "Rendered Image",
        size: float = 3.5
    ):
        """Initializes the ImageWindow.

        Args:
            master (tk.Tk): The parent window.
            title (str, optional): The title of the window. Defaults to "Rendered Image".
            size (float, optional): Ratio to resize the image. Defaults to 3.5.
        """
        self.master = master
        self.master.title(title)
        self.label = tk.Label(self.master)
        self.label.pack()
        self.ratio = size

    @ignore_unhashable
    @lru_cache(maxsize=10)
    def update_image(self, image: list[float]) -> None:
        """Updates the image displayed in the window.
        Has a cache to prevent unnecessary resizing and rendering.

        Args:
            image (list[float]): The image to display.
        """
        width, height, *_ = image.shape
        image = Image.fromarray(image)
        image = image.resize(
            (int(width * self.ratio), int(height * self.ratio)),
            Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(image)
        self.label.configure(image=photo)
        self.label.image = photo
