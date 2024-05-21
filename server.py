"""Server module for the AI Festival experience."""
from queue import Queue
from typing import Union
import json
import os
from os import listdir
import pickle
import random
import socket
import shutil
import threading
import time
import tkinter as tk

from pynput import keyboard
from pynput.keyboard import Key
from PIL import Image
import numpy as np
import pygame

from render import ImageWindow
from utils import ACTIONS_MAPPING, Connection
from utils import create_environment


class Server:
    """Server class for the AI Festival experience.

    Parameters:
        HOST: The IP address of the server.
        PORT: The port of the server.
        BUFFER_SIZE: The size of the buffer for receiving data.

        s (socket.socket): socket connection
        done (bool): whether the game is done
        human (bool): whether the player is human
        pressed_keys (list): list of pressed keys - used for actions
        closing (bool): whether the server is closing
        record (bool): whether to record the experience
        episode (int): episode number (look at root_dir and increments by 1)
        timestep (int): timestep number
        actions (list): list of actions
        status (dict): status of the episode (True if got to the end, False if died)
        root_dir (str): root directory for the recordings
        environment (gym.Env): gym environment
        root (tk.Tk): tkinter root
        app (ImageWindow): image window
        threads (list): list of threads (connect, render_frame, step)
        listener (keyboard.Listener): keyboard listener
        frame (np.ndarray): frame of the environment
    """

    HOST = "10.70.255.242"
    PORT = 16006
    BUFFER_SIZE = 1024

    def __init__(self, env_name: str = "SuperMarioBros-1-1-v0", record: bool = False):
        """Server class for the AI Festival experience.

        Args:
            env_name (str, optional): gym environment name. Defaults to "SuperMarioBros-1-1-v0".
            record (bool, optional): whether to record the experience. Defaults to False.
        """
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.done = False
        self.human = True
        self.pressed_keys = []
        self.closing = False
        self.connection_type = Connection.ACTION
        if self.connection_type == Connection.ACTION:
            self.action_queue = Queue()

        self.record = record
        self.episode = 0
        self.timestep = 0
        self.actions = []
        self.status = {}
        self.root_dir = "./tmp/recordings/"
        self.timeout = 1/40
        if self.record:
            self.start_recording()

        self.environment = create_environment(env_name)
        self.reset()

        self.root = tk.Tk()
        self.app = ImageWindow(self.root, "Player")

        self.open_socket()

        self.threads = []
        thread = threading.Thread(target=self.connect)
        thread.start()
        self.threads.append(thread)

        thread = threading.Thread(target=self.render_frame)
        thread.start()
        self.threads.append(thread)

        thread = threading.Thread(target=self.step)
        thread.start()
        self.threads.append(thread)

        thread = threading.Thread(target=self.listen_joypad)
        thread.start()
        self.threads.append(thread)

        self.listen_keyboard()

        self.root.mainloop()
        self.root.update()
        exit()

    def load_replay(self) -> list[list[float]]:
        path = "./tmp/agent_play/"
        folders = [
            os.path.join(path, f)
            for f in listdir(path)
            if ".ipynb_checkpoints" not in f
        ]
        folder = random.choice(folders)
        images = [
            os.path.join(folder, f)
            for f in listdir(folder)
            if "png" in f
        ]
        images.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        return images, folder.split("/")[-1]

    ##################### SOCKET RELATED #####################
    def open_socket(self) -> None:
        """Open socket connection."""
        print(f"Connection opened on {self.HOST} at {self.PORT}")
        self.s.bind((self.HOST, self.PORT))
        self.s.listen()

    def connect(self) -> None:
        """Connect to the client."""
        self.conn, self.addr = self.s.accept()

        self.human = True
        if random.random() > 1:
            print("agent")
            self.human = False
            self.agent_replay_count = 0
            self.agent_replay, self.folder = self.load_replay()
        else:
            print("human")

        print(f"Connected by {self.addr}")
        self.reset()
        while not self.closing:
            data = self.conn.recv(self.BUFFER_SIZE)
            data = json.loads(data.decode())
            match data.get("action", ""):
                case "frame":
                    if self.human:
                        if self.connection_type == Connection.FRAME:
                            if self.done:
                                data = {"status": "finish", "human": True}
                                self.conn.send(json.dumps(data).encode())
                            else:
                                self.send_frame()
                        else:
                            if self.done and self.action_queue.empty():
                                data = {"status": "finish", "human": True}
                                self.conn.send(json.dumps(data).encode())
                            else:
                                action = self.action_queue.get()
                                data = {"human": True, "action": action}
                                self.conn.send(json.dumps(data).encode())
                    else:
                        if self.agent_replay_count == len(self.agent_replay) - 1:
                            data = {"status": "finish", "human": self.human}
                            self.conn.send(json.dumps(data).encode())
                        else:
                            if self.connection_type == Connection.FRAME:
                                self.send_replay()
                            else:
                                time.sleep(self.timeout)
                                data = {
                                    "human": False,
                                    "recording": self.folder,
                                    "index": self.agent_replay_count
                                }
                                self.conn.send(json.dumps(data).encode())
                                self.agent_replay_count += 1

                case "close":
                    print(f"Closing connection with {self.addr}")
                    self.conn.send(b"ok")
                    self.conn.close()
                    break
                case _:
                    self.conn.send(b"{}")
        self.connect()

    def close(self) -> None:
        """Verifies data, closes all connections, and terminate all threads."""
        self.verify_data()
        for thread in self.threads:
            thread.join(0)
        self.root.destroy()
        self.root.quit()
        self.s.close()
        self.listener.stop()
        exit()

    def send_frame(self) -> None:
        """Send frame to the client."""
        h, w, c = self.frame.shape
        render = self.frame.reshape((-1))
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

    def send_replay(self) -> None:
        """Send replay to the client."""
        frame = self.agent_replay[self.agent_replay_count]
        frame = Image.open(frame)
        frame = np.array(frame)
        h, w, c = frame.shape
        render = frame.reshape((-1))
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
        if self.agent_replay_count + 1 < len(self.agent_replay):
            self.agent_replay_count += 1

    ##################### INPUT RELATED #####################
    def listen_keyboard(self) -> None:
        """Listen to keyboard inputs."""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def listen_joypad(self) -> None:
        """Listen to joypad inputs."""
        print("Start listening to joypad")

        pygame.init()
        if pygame.joystick.get_count() == 0:
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.last_horizontal = None
        self.last_vertical = None

        while not self.closing:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    self.on_joy_press(event.button)
                if event.type == pygame.JOYBUTTONUP:
                    self.on_joy_release(event.button)
                if event.type == pygame.JOYAXISMOTION:
                    if event.axis == 0:
                        # left and right
                        match int(event.value):
                            case 1:
                                self.on_joy_press("right")
                                self.last_horizontal = "right"
                            case 0:
                                self.on_joy_release(self.last_horizontal)
                                self.last_horizontal = None
                            case -1:
                                self.on_joy_press("left")
                                self.last_horizontal = "left"
                    else:
                        # up and down
                        match int(event.value):
                            case -1:
                                self.on_joy_press("up")
                                self.last_vertical = "up"
                            case 0:
                                self.on_joy_release(self.last_vertical)
                                self.last_vertical = None
                            case 1:
                                self.on_joy_press("down")
                                self.last_vertical = "down"

    def on_joy_press(self, key: str) -> None:
        match key:
            case 1:
                self.add_pressed_keys("A")
            case 0:
                self.add_pressed_keys("B")
            case "right":
                self.add_pressed_keys("right")
            case "left":
                self.add_pressed_keys("left")
            case "up":
                self.add_pressed_keys("up")
            case "down":
                self.add_pressed_keys("down")
            case _:
                self.add_pressed_keys("NOOP")

    def on_joy_release(self, key: str) -> None:
        match key:
            case 1:
                self.remove_pressed_keys("A")
            case 0:
                self.remove_pressed_keys("B")
            case "right":
                self.remove_pressed_keys("right")
            case "left":
                self.remove_pressed_keys("left")
            case "up":
                self.remove_pressed_keys("up")
            case "down":
                self.remove_pressed_keys("down")
            case _:
                self.remove_pressed_keys("NOOP")

    def get_key(self, key: Key) -> str:
        """Get key from the event.

        Args:
            key (Key): key event

        Returns:
            str: key
        """
        try:
            return key.char
        except AttributeError:
            pass
        return key

    def on_press(self, key: str) -> None:
        """Handle key press event.

        Args:
            key (str): key string
        """
        match self.get_key(key):
            case keyboard.Key.up:
                self.add_pressed_keys("up")
            case keyboard.Key.down:
                self.add_pressed_keys("down")
            case keyboard.Key.right:
                self.add_pressed_keys("right")
            case keyboard.Key.left:
                self.add_pressed_keys("left")
            case "z":
                self.add_pressed_keys("A")
            case "x":
                self.add_pressed_keys("B")
            case "r":
                self.frame = self.environment.reset()
            case _:
                self.add_pressed_keys("NOOP")

    def on_release(self, key: str) -> Union[None, bool]:
        """Handle key release event.

        Args:
            key (str): key string

        Returns:
            Union[None, bool]: False if closing, None otherwise
        """
        match self.get_key(key):
            case keyboard.Key.up:
                self.remove_pressed_keys("up")
            case keyboard.Key.down:
                self.remove_pressed_keys("down")
            case keyboard.Key.right:
                self.remove_pressed_keys("right")
            case keyboard.Key.left:
                self.remove_pressed_keys("left")
            case "z":
                self.remove_pressed_keys("A")
            case "x":
                self.remove_pressed_keys("B")
            case keyboard.Key.esc:
                self.closing = True
                self.close()
                return False
            case _:
                self.remove_pressed_keys("NOOP")

    def add_pressed_keys(self, key: str) -> None:
        """Add pressed keys to the list. Remove duplicates and sort for correct mapping.

        Args:
            key (str): key string
        """
        self.pressed_keys.append(key)
        self.pressed_keys = list(set(self.pressed_keys))
        self.pressed_keys.sort()

    def remove_pressed_keys(self, key: str) -> None:
        """Remove pressed keys from the list.

        Args:
            key (str): key string
        """
        try:
            self.pressed_keys.remove(key)
        except ValueError:
            pass

    def get_action_from_pressed_keys(self) -> int:
        """Get action from pressed keys.

        Returns:
            int: Action from ACTIONS_MAPPING dictionary, 0 otherwise
        """
        return ACTIONS_MAPPING.get(tuple(self.pressed_keys), 0)

    ##################### GYM RELATED #####################
    def start_recording(self) -> None:
        """Start recording the experience."""
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            self.episode = 0
        else:
            episodes = [int(f) for f in listdir(self.root_dir) if ".pkl" not in f]
            if len(episodes) == 0:
                self.episode = 0
            else:
                self.episode = max(episodes) + 1

        if not os.path.exists(f"{self.root_dir}status.pkl"):
            self.status = {}
        else:
            with open(f"{self.root_dir}status.pkl", "rb") as status:
                self.status = pickle.load(status)

    def render_frame(self) -> None:
        """Render the frame."""
        while not self.closing:
            self.app.update_image(self.frame)

    def step(self) -> None:
        """Step through the environment."""
        while not self.closing:
            try:
                action = self.get_action_from_pressed_keys()

                self.frame, reward, done, truncated, info = self.environment.step(action)
                done |= truncated
                self.done = done
                if self.connection_type == Connection.ACTION:
                    self.action_queue.put(action)

                if self.record:
                    if done and info["flag_get"]:
                        self.status[self.episode] = True
                        self.save_status()
                    elif done:
                        self.status[self.episode] = False
                        self.save_status()

                    self.save_image()
                    self.actions.append(action)
                    self.timestep += 1
            except ValueError:
                if self.record:
                    self.save_actions()
                self.episode += 1
                self.timestep = 0
            time.sleep(self.timeout)

    def reset(self) -> None:
        """Reset the environment."""
        self.frame = self.environment.reset()
        self.done = False

        if self.connection_type == Connection.ACTION:
            self.action_queue = Queue()

        if self.record:
            if not os.path.exists(f"{self.root_dir}{self.episode}/"):
                os.makedirs(f"{self.root_dir}{self.episode}")
            self.save_image()
            self.actions = []

    def save_image(self) -> None:
        """Save the state image."""
        Image.fromarray(self.frame).save(
            f"{self.root_dir}{self.episode}/{self.timestep}.png")

    def save_actions(self) -> None:
        """Save the actions."""
        with open(f"{self.root_dir}{self.episode}/action.pkl", "wb") as f:
            pickle.dump(self.actions, f)

    def save_status(self) -> None:
        """Save the status (got to the end or died)."""
        with open(f"{self.root_dir}status.pkl", "wb") as f:
            pickle.dump(self.status, f)

    def verify_data(self) -> None:
        """Verify the data and delete the corrupted ones."""
        to_be_deleted = []
        folders = list(next(os.walk(f"{self.root_dir}")))[1]
        for folder in folders:
            if not os.path.exists(f"{self.root_dir}{folder}/action.pkl"):
                print(f"Deleting: {self.root_dir}{folder} - no action found")
                shutil.rmtree(f"{self.root_dir}{folder}")
                to_be_deleted.append(int(folder))
                continue

            with open(f"{self.root_dir}{folder}/action.pkl", "rb") as f:
                actions = pickle.load(f)
                images = [
                    f for f in listdir(f"{self.root_dir}{folder}")
                    if "pkl" not in f
                ]
                if len(actions) != len(images):
                    print(f"Deleting: {self.root_dir}{folder} - length mismatch")
                    shutil.rmtree(f"{self.root_dir}{folder}")
                    to_be_deleted.append(int(folder))
                    continue

        status = {}
        for key, value in self.status.items():
            if key not in to_be_deleted:
                status[key] = value
        self.status = status
        self.save_status()


if __name__ == "__main__":
    server = Server(record=False)
