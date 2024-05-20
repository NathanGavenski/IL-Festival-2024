"""Module for the client to request and display rendered images."""
import json
import socket
import threading
import tkinter as tk

import numpy as np
from PIL import Image

from render import ImageWindow
from utils import Connection
from utils import create_environment


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
        self.buttons = False
        self.connection_type = Connection.FRAME
        self.recording_path = "./tmp/agent_play/"

        if self.connection_type == Connection.ACTION:
            self.env = create_environment("SuperMarioBros-1-1-v0")
            self.app.update_image(self.env.reset())

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
            self.frame = self.request_frame()
            try:
                self.app.update_image(self.frame)
            except RuntimeError:
                self.close()
                break
            except AttributeError:
                self.close()
                break

    def display_options(self, label: str) -> None:
        """"""
        if not self.buttons:
            self.buttons = True
            self.app.add_buttons(label, self.close)

    def request_frame(self) -> list[float]:
        """Requests a frame from the server.

        Returns:
            list[float]: The rendered image.
        """
        data = json.dumps({"action": "frame"})
        self.s.send(data.encode())

        if self.connection_type == Connection.FRAME:
            frame = []
            index = 0
            length = np.inf
            h, w, c = 0, 0, 0
            while index < length:
                response = self.get_response()
                if "status" in response.keys():
                    self.display_options(response.get("human"))
                    continue

                frame += response.get("frame", [])
                index = response.get("index", 0)
                length = response.get("length", 0)
                data = json.dumps({"index": index, "length": length})
                self.s.send(data.encode())

            h = response.get("info", {}).get("height", 400)
            w = response.get("info", {}).get("width", 600)
            c = response.get("info", {}).get("channels", 3)
            frame = np.array(frame).reshape((h, w, c))
        else:
            response = self.get_response()
            if response.get("human"):
                if "status" in response.keys():
                    self.display_options(response.get("human"))
                    return self.frame
                else:
                    action = response.get("action")
                    try:
                        frame, *_ = self.env.step(action)
                    except ValueError:
                        return self.frame
            else:
                if "status" in response.keys():
                    self.display_options(response.get("human"))
                    return self.frame
                else:
                    recording = response.get("recording")
                    index = response.get("index")
                    path = f"{self.recording_path}{recording}/{index}.png"
                    frame = np.array(Image.open(path))
        return frame.astype("uint8")

    def get_response(self) -> dict[str, any]:
        try:
            response = self.s.recv(self.BUFFER_SIZE)
        except ConnectionResetError:
            self.close(True)
            return

        return json.loads(response.decode())


if __name__ == "__main__":
    client = Client()
