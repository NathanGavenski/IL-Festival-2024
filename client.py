import json
import queue
import socket
import threading
import tkinter as tk

import numpy as np

from render import ImageWindow


class Client:

    HOST = "127.0.0.1"
    PORT = 16006
    BUFFER_SIZE = 8192

    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.root = tk.Tk()
        self.app = ImageWindow(self.root, "Viewer")

        self.connect()

        self.update_queue = queue.Queue()
        thread = threading.Thread(target=self.display_frame)
        thread.start()

        self.root.mainloop()
        self.update_queue.get()

        self.close()

    def connect(self) -> None:
        self.s.connect((self.HOST, self.PORT))

    def close(self) -> None:
        self.s.close()

    def display_frame(self) -> None:
        while True:
            frame = self.request_frame()
            try:
                self.app.update_image(frame)
            except RuntimeError:
                self.update_queue.put(None)
                break

    def request_frame(self) -> list[float]:
        data = json.dumps({"action": "frame"})
        self.s.send(data.encode())

        frame = []
        index = 0
        length = np.inf
        h, w, c = 0, 0, 0
        while index < length:
            response = self.s.recv(self.BUFFER_SIZE)
            response = json.loads(response.decode())
            frame += response.get("frame", [])
            index = response.get("index", 0)
            length = response.get("length", 0)
            data = json.dumps({"index": index, "length": length})
            self.s.send(data.encode())

        h = response.get("info", {}).get("height", 400)
        w = response.get("info", {}).get("width", 600)
        c = response.get("info", {}).get("channels", 3)
        frame = np.array(frame).reshape((h, w, c))

        return frame.astype("uint8")


if __name__ == "__main__":
    client = Client()
