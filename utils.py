from typing import Any, Callable
import functools
from enum import Enum

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import StepAPICompatibility, TimeLimit


class Connection(Enum):
    FRAME = 1
    ACTION = 2


ACTIONS_MAPPING = {
    ('NOOP'): 0,
    ('right', ): 1,
    ('A', 'right', ): 2,
    ('B', 'right', ): 3,
    ('A', 'B', 'right', ): 4,
    ('A', ): 5,
    ('B', ): 6,
    ('left', ): 7,
    ('A', 'left', ): 8,
    ('B', 'left', ): 9,
    ('A', 'B', 'left', ): 10,
    ('down', ): 11,
    ('up', ): 12,
    ('A', 'B', ): 13,
}

ACTIONS = [
    ['NOOP'],
    ['right'],
    ['A', 'right'],
    ['B', 'right'],
    ['A', 'B', 'right'],
    ['A'],
    ['B'],
    ['left'],
    ['A', 'left'],
    ['B', 'left'],
    ['A', 'B', 'left'],
    ['down'],
    ['up'],
    ['A', 'B']
]


def ignore_unhashable(func: Callable[Any, tuple]) -> tuple[int]:
    """ Ignore unhashable types in the cache

    Args:
        func (Callable[Any, tuple]): wrapped function to ignore unhashable types

    Returns:
        tuple[int]: tuple of hashable types
    """
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ('cache_info', 'cache_clear')

    @functools.wraps(func, assigned=attributes)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if 'unhashable type' in str(error):
                return uncached(*args, **kwargs)
            raise
    wrapper.__uncached__ = uncached
    return wrapper


def create_environment(env_name: str) -> gym.Env:
    """Create the gym environment.

    Args:
        env_name (str): gym environment name

    Returns:
        gym.Env: gym environment
    """
    env = gym.make(env_name)
    steps = env._max_episode_steps

    env = JoypadSpace(env.env, ACTIONS)

    def gymnasium_reset(self, **kwargs):
        return self.env.reset()
    env.reset = gymnasium_reset.__get__(env, JoypadSpace)

    env = StepAPICompatibility(env, output_truncation_bool=True)
    env = TimeLimit(env, max_episode_steps=steps)
    return env
