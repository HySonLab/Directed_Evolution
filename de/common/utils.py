import os
import numpy as np
import time
import torch
import random
from datetime import datetime
from functools import wraps
from typing import List


def split_kmers2(seqs: List[str], k: int = 3) -> List[List[str]]:
    return [[seq[i:i + k] for i in range(len(seq) - k + 1)] for seq in seqs]


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed (int): seed number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_full_deterministic(seed: int):
    """Helper function for reproducible behavior during distributed training
    See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    set_seed(seed)

    # Enable PyTorch deterministic mode. This potentially requires either the environment
    # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Enable CuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_variant_in_color(seq: str,
                           wt: str,
                           ignore_gaps: bool = True) -> None:
    """Print a variant in color."""
    for j in range(len(wt)):
        if seq[j] != wt[j]:
            if ignore_gaps and (seq[j] == '-' or seq[j] == 'X'):
                continue
            print(f'\033[91m{seq[j]}', end='')
        else:
            print(f'\033[0m{seq[j]}', end='')
    print('\033[0m')


def timer(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'{now}: Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
