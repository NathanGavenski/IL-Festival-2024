import json
import socket
import threading
import tkinter as tk

import gymnasium as gym

from render import ImageWindow


class Server:

    HOST = "127.0.0.1"
    PORT = 16006
    BUFFER_SIZE = 1024

    def __init__(self, env_name: str = "ALE/DonkeyKong-v5"):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.environment = self.create_environment(env_name)
        self.state, _ = self.environment.reset()
        self.done = False

        self.root = tk.Tk()
        self.app = ImageWindow(self.root, "Player")

        self.open_socket()

        thread = threading.Thread(target=self.connect)
        thread.start()

        thread = threading.Thread(target=self.render_frame)
        thread.start()

        self.root.mainloop()

    ##################### SOCKET RELATED #####################
    def open_socket(self) -> None:
        print(f"Connection opened on {self.HOST} at {self.PORT}")
        self.s.bind((self.HOST, self.PORT))
        self.s.listen()

    def connect(self) -> None:
        self.conn, self.addr = self.s.accept()
        print(f"Connected by {self.addr}")
        while True:
            data = self.conn.recv(self.BUFFER_SIZE)
            if not data:
                break

            data = json.loads(data.decode())
            match data.get("action", ""):
                case "frame":
                    self.send_frame()
                case "close":
                    self.conn.send(b"ok")
                    break
                case _:
                    self.conn.send(b"")
        self.connect()

    def close(self) -> None:
        print("Connection closed")
        self.s.close()

    def send_frame(self) -> None:
        render = self.environment.render()
        h, w, c = render.shape
        render = render.reshape((-1))
        indexes = list(range(0, render.shape[0], 1024))
        indexes += [render.shape[0]]
        for i, index in enumerate(indexes[:-1]):
            data = {
                "info": {
                    "height": h,
                    "width": w,
                    "channels": c
                },
                "frame": render[index:indexes[i+1]].tolist(),
                "index": i,
                "length": len(indexes) - 2,
            }
            data = json.dumps(data)
            self.conn.send(data.encode())
            self.conn.recv(self.BUFFER_SIZE)

    ##################### GYM RELATED #####################
    def render_frame(self) -> None:
        while True:
            frame = self.environment.render()
            self.app.update_image(frame)

    def create_environment(self, env_name: str) -> gym.Env:
        return gym.make(env_name, render_mode="rgb_array")


if __name__ == "__main__":
    server = Server()
