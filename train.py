from functools import partial
import pickle
from os import listdir
import os
from os.path import join
from typing import Callable
import types

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from benchmark.methods import BC
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import StepAPICompatibility, TimeLimit
from PIL import Image
import torch
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tensorboard_wrapper.tensorboard import Tensorboard


from utils import ACTIONS


def create_env() -> gym.Env:
    env = gym.make("SuperMarioBros-1-1-v0")
    steps = env._max_episode_steps

    env = JoypadSpace(env.env, ACTIONS)

    def gymnasium_reset(self, **kwargs):
        return self.env.reset()
    env.reset = gymnasium_reset.__get__(env, JoypadSpace)

    env = StepAPICompatibility(env, output_truncation_bool=True)
    env = TimeLimit(env, max_episode_steps=steps)
    return env


def enjoy(self, transforms: Callable[[Tensor], Tensor] = None) -> dict[str, float]:
    env = create_env()
    average_reward = []
    for _ in tqdm(range(100)):
        done = False
        obs = env.reset()
        accumulated_reward = 0
        while not done:
            if transforms is not None:
                obs = transforms(obs)
                obs = obs[None]
            action = self.predict(obs)
            obs, reward, done, truncated, _ = env.step(action.item())
            done |= truncated
            accumulated_reward += reward
        average_reward.append(accumulated_reward)
    return {"aer": sum(average_reward) / 100}

def train(
    self,
    n_epochs: int,
    train_dataset: DataLoader,
    eval_dataset: DataLoader = None,
    always_save: bool = False,
) -> Self:
    """Train process.

    Args:
        n_epochs (int): amount of epoch to run.
        train_dataset (DataLoader): data to train.
        eval_dataset (DataLoader): data to eval. Defaults to None.

    Returns:
        method (Self): trained method.
    """
    folder = f"./benchmark_results/bc/{self.environment_name}"
    if not os.path.exists(folder):
        os.makedirs(f"{folder}/")

    board = Tensorboard(path=folder)
    board.add_hparams(self.hyperparameters)
    self.policy.to(self.device)

    best_model = -np.inf

    pbar = range(n_epochs)
    if self.verbose:
        pbar = tqdm(pbar, desc=self.__method_name__)
    for epoch in pbar:
        train_metrics = self._train(train_dataset)
        board.add_scalars("Train", epoch="train", **train_metrics)

        if eval_dataset is not None:
            eval_metrics = self._eval(eval_dataset)
            board.add_scalars("Eval", epoch="eval", **eval_metrics)
            board.step(["train", "eval"])
        else:
            board.step("train")

        if train_metrics["accuracy"] > best_model:
            best_model = train_metrics["accuracy"]
            self.save(name=epoch if always_save else None)

    return self


class MarioDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform: Callable[[Tensor], Tensor] = None
    ) -> None:
        self.path = path
        if isinstance(path, list):
            self.states, self.actions = self.load_data(path[0])
            for p in path[1:]:
                states, actions = self.load_data(p)
                self.states += states
                self.actions = torch.cat((self.actions, actions), dim=0)
        else:
            self.states, self.actions = self.load_data(path)

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def load_data(self, path: str) -> tuple[Tensor, Tensor]:
        actions = pickle.load(open(f"{path}action.pkl", "rb"))
        actions = torch.tensor(actions)
        images = [join(path, f) for f in listdir(path) if "png" in f]
        images.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        return images, actions

    def __len__(self) -> int:
        return self.actions.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        state = self.states[idx]
        action = torch.tensor([self.actions[idx]])

        state = Image.open(state)
        state = self.transform(state)
        return state, action, torch.tensor([])


if __name__ == "__main__":
    dataset = MarioDataset([
        "./tmp/recordings/1/",
        "./tmp/recordings/4/"
    ])
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    env = create_env()
    bc = BC(env, config_file="./bc.yaml", verbose=True, enjoy_criteria=999999)
    enjoy = partial(
        enjoy,
        transforms=transforms.Compose([
            Image.fromarray,
            transforms.ToTensor(),
        ])
    )
    bc._enjoy = types.MethodType(enjoy, bc)
    bc.train = types.MethodType(train, bc)

    bc.train(
        n_epochs=1000,
        train_dataset=dataloader
    )
    print(bc._enjoy())
