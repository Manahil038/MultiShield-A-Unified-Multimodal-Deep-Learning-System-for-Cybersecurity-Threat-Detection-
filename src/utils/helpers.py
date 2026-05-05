import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for full reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
