"""Module for the client to request and display rendered images."""
import json
import socket
import threading
import tkinter as tk

import numpy as np

from render import ImageWindow


class Client:
    """A client to request and display rendered images.

    Parameters:
        HOST: The IP address of the server.
        PORT: The port of the server.
        BUFFER_SIZE: The size of the buffer for receiving data.
        
        s: The socket to communicate with the server.
        root: The main window of the client.
        app: The window to display the rendered images.
        threads: The threads to run the client.
    """

    HOST = "127.0.0.1"
    PORT = 16006
    BUFFER_SIZE = 8192

    def __init__(self):
        """Initializes the client."""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.root = tk.Tk()
        self.app = ImageWindow(self.root, "Viewer")

        self.connect()

        self.threads = []
        thread = threading.Thread(target=self.display_frame)
        thread.start()
        self.threads.append(thread)

        self.root.mainloop()
        self.root.update()

    def connect(self) -> None:
        """Connects to the server."""
        self.s.connect((self.HOST, self.PORT))

    def close(self, server: bool = False) -> None:
        """Closes the connection and stops the client.

        Args:
            server (bool, optional): Whether the server closed the connection. Defaults to False.
        """
        if not server:
            print("Closing connection")
            data = json.dumps({"action": "close"})
            self.s.send(data.encode())
            self.s.close()

        print("stopping threads")
        for thread in self.threads:
            try:
                thread.join(0)
            except RuntimeError:
                pass

        print("stopping interface")
        self.root.destroy()
        self.root.quit()
        exit()

    def display_frame(self) -> None:
        """Displays the rendered images."""
        while True:
            frame = self.request_frame()
            try:
                self.app.update_image(frame)
            except RuntimeError:
                self.close()
                break
            except AttributeError:
                self.close()
                break

    def request_frame(self) -> list[float]:
        """Requests a frame from the server.

        Returns:
            list[float]: The rendered image.
        """
        data = json.dumps({"action": "frame"})
        self.s.send(data.encode())

        frame = []
        index = 0
        length = np.inf
        h, w, c = 0, 0, 0
        while index < length:
            try:
                response = self.s.recv(self.BUFFER_SIZE)
            except ConnectionResetError:
                self.close(True)
                return

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
