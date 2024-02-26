import json
import os
import pickle
import random
import socket
import threading
import time
import tkinter as tk

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import StepAPICompatibility, TimeLimit
from pynput import keyboard
from PIL import Image

from render import ImageWindow
from utils import ACTIONS, ACTIONS_MAPPING


class Server:

    HOST = "127.0.0.1"
    PORT = 16006
    BUFFER_SIZE = 1024

    def __init__(self, env_name: str = "SuperMarioBros-1-1-v0", record: bool = False):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.done = False
        self.human = True
        self.pressed_keys = []
        self.closing = False

        self.record = record
        self.episode = 0
        self.timestep = 0
        self.actions = []
        self.root_dir = "./tmp/recordings/"
        if self.record:
            self.start_recording()

        self.environment = self.create_environment(env_name)
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

        self.listen_keyboard()

        self.root.mainloop()
        self.root.update()
        exit()

    ##################### SOCKET RELATED #####################
    def open_socket(self) -> None:
        print(f"Connection opened on {self.HOST} at {self.PORT}")
        self.s.bind((self.HOST, self.PORT))
        self.s.listen()

    def connect(self) -> None:
        self.conn, self.addr = self.s.accept()

        self.human = True
        if random.random() > 1:
            self.human = False

        print(f"Connected by {self.addr}")
        while not self.closing:
            data = self.conn.recv(self.BUFFER_SIZE)
            data = json.loads(data.decode())
            match data.get("action", ""):
                case "frame":
                    if self.human:
                        self.send_frame()
                    else:
                        # send agent replay
                        pass
                case "close":
                    print(f"Closing connection with {self.addr}")
                    self.conn.send(b"ok")
                    self.conn.close()
                    break
                case _:
                    self.conn.send(b"{}")
        self.connect()

    def close(self) -> None:
        self.closing = True
        self.s.close()
        self.listener.stop()
        for thread in self.threads:
            thread.join(0)
        self.root.destroy()
        self.root.quit()
        exit()

    def send_frame(self) -> None:
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

    ##################### INPUT RELATED #####################
    def listen_keyboard(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def get_key(self, key):
        try:
            return key.char
        except AttributeError:
            pass
        return key

    def on_press(self, key) -> None:
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

    def on_release(self, key) -> None:
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
                self.close()
                return False
            case _:
                self.remove_pressed_keys("NOOP")

    def add_pressed_keys(self, key) -> None:
        self.pressed_keys.append(key)
        self.pressed_keys = list(set(self.pressed_keys))
        self.pressed_keys.sort()

    def remove_pressed_keys(self, key) -> None:
        try:
            self.pressed_keys.remove(key)
        except ValueError:
            pass

    def get_action_from_pressed_keys(self) -> int:
        return ACTIONS_MAPPING.get(tuple(self.pressed_keys), 0)

    ##################### GYM RELATED #####################
    def start_recording(self) -> None:
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            self.episode = 0
        else:
            self.episode = len(next(os.walk(self.root_dir))[1])

    def render_frame(self) -> None:
        while not self.closing:
            self.app.update_image(self.frame)

    def create_environment(self, env_name: str) -> gym.Env:
        env = gym.make(env_name)
        steps = env._max_episode_steps

        env = JoypadSpace(env.env, ACTIONS)
        def gymnasium_reset(self, **kwargs):
            return self.env.reset()
        env.reset = gymnasium_reset.__get__(env, JoypadSpace)

        env = StepAPICompatibility(env, output_truncation_bool=True)
        env = TimeLimit(env, max_episode_steps=steps)
        return env

    def step(self) -> None:
        while not self.closing:
            try:
                action = self.get_action_from_pressed_keys()
                self.frame, *_ = self.environment.step(action)
                if self.record:
                    self.save_image()
                    self.actions.append(action)
                    self.timestep += 1
            except ValueError:
                self.save_actions()
                self.episode += 1
                self.timestep = 0
                self.reset()
            time.sleep(1/40)

    def reset(self) -> None:
        self.frame = self.environment.reset()

        if self.record:
            if not os.path.exists(f"{self.root_dir}{self.episode}/"):
                os.makedirs(f"{self.root_dir}{self.episode}")
            self.save_image()

    def save_image(self) -> None:
        Image.fromarray(self.frame).save(f"{self.root_dir}{self.episode}/{self.timestep}.png")

    def save_actions(self) -> None:
        with open(f"{self.root_dir}{self.episode}/action.pkl", "wb") as f:
            pickle.dump(self.actions, f)


if __name__ == "__main__":
    server = Server(record=True)
