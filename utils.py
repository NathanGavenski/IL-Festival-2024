from typing import Any, Callable
import functools


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
