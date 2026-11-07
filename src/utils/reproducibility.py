"""Reproducibility utilities for fixing seeds across frameworks."""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python standard library, NumPy, and environment.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional torch/lightgbm seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

