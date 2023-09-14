import os
import numpy as np
import time
import torch
import random
from datetime import datetime
from functools import wraps
from typing import List


def split_kmers(seqs: List[str], k: int = 3) -> List[List[str]]:
    return [[seq[i:i + k] for i in range(0, len(seq), k)] for seq in seqs]


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


def save_vocabulary(save_dir: str,
                    standard_tokens: List[str],
                    k: int,
                    prepend_tokens: List[str] = ["<cls>", "<pad>", "<eos>", "<unk>"],
                    append_tokens: List[str] = ["<mask>"]) -> List[str]:
    # Check if vocab file exists
    filepath = os.path.join(save_dir, f"vocab_{k}.txt")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            all_tokens = [line.rstrip() for line in f]
        return all_tokens

    all_tokens = []
    all_tokens.extend(prepend_tokens)
    all_tokens.extend(standard_tokens)
    for i in range((8 - (len(all_tokens) % 8)) % 8):
        all_tokens.append(f"<null_{i  + 1}>")
    all_tokens.extend(append_tokens)

    os.makedirs(save_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write('\n'.join(all_tokens))

    return all_tokens


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
