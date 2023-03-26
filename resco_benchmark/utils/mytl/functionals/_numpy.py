import contextlib
from typing import Optional

import numpy as np


@contextlib.contextmanager
def set_local_seed(seed: Optional[int] = None):
    """Set a fixed numpy seed locally."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
