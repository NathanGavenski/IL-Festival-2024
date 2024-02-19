import tkinter as tk
from PIL import Image, ImageTk


class ImageWindow:
    def __init__(self, master: tk.Tk, title: str = "Rendered Image"):
        self.master = master
        self.master.title(title)
        self.label = tk.Label(self.master)
        self.label.pack()

    def update_image(self, image: list[float]) -> None:
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.label.configure(image=photo)
        self.label.image = photo
