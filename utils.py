from typing import Any, Callable
import functools


ACTIONS_MAPPING = {
    ('NOOP'): 0,
    ('right', ): 1,
    ('A', 'right', ): 2,
    ('B', 'right', ): 3,
    ('A', 'B', 'right', ): 4,
    ('A', ): 5,
    ('left', ): 6,
    ('A', 'left', ): 7,
    ('B', 'left', ): 8,
    ('A', 'B', 'left', ): 9,
    ('down', ): 10,
    ('up', ): 11,
    ('A', 'B', ): 12,
}

ACTIONS = [
    ['NOOP'],
    ['right'],
    ['A', 'right'],
    ['B', 'right'],
    ['A', 'B', 'right'],
    ['A'],
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
