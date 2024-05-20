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

        self.button_frame = tk.Frame(self.master, bg="white")
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.ratio = size

    def add_buttons(self, human: bool, callback: callable) -> None:
        """Adds 'Human' and 'Agent' buttons on top of the image."""
        button_font = ('Helvetica', 14)  # Define the font and size for the buttons
        button_width = 15  # Define the width of the buttons
        button_height = 2  # Define the height of the buttons
        self.ground_truth = human
        self.callback = callback

        self.human_button = tk.Button(
            self.button_frame,
            text="Human",
            font=button_font,
            width=button_width,
            height=button_height,
            bg="lightgreen",
            fg="black",
            command=self.on_human_press
        )
        self.agent_button = tk.Button(
            self.button_frame,
            text="Agent",
            font=button_font,
            width=button_width,
            height=button_height,
            bg="lightgreen",
            fg="black",
            command=self.on_agent_press
        )
        self.human_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        self.agent_button.pack(side=tk.RIGHT, padx=5, pady=5, expand=True)

    def on_human_press(self) -> None:
        text = "Correct" if self.ground_truth else "Incorrect"
        correct_label = tk.Label(self.master, text=text, font=("Helvetica", 36))
        correct_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        close_button = tk.Button(self.master, text="Close", command=self.callback)
        close_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def on_agent_press(self) -> None:
        text = "Correct" if not self.ground_truth else "Incorrect"
        correct_label = tk.Label(self.master, text=text, font=("Helvetica", 36))
        correct_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        close_button = tk.Button(self.master, text="Close", command=self.callback)
        close_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

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
