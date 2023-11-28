import os
import numpy as np
import time
import torch
import random
from datetime import datetime
from functools import wraps
from polyleven import levenshtein
from typing import List
from .constants import CANONICAL_ALPHABET


def get_mutated_sequence(focus_seq: str,
                         mutant: str,
                         start_idx: int = 1,
                         AA_vocab: str = ''.join(CANONICAL_ALPHABET)) -> str:
    """Mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).

    Args:
        focus_seq (str): Input sequence.
        mutant (str): list of mutants applied to input sequence (e.g., "B12F:A83M").
        start_idx (int): Index to start indexing.
        AA_vocab (str): Amino acids.

    Returns:
        (str): mutated sequence.
    """
    if mutant == "":
        return focus_seq
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(
                mutation[1:-1]), mutation[-1]
        except ValueError:
            print("Issue with mutant: " + str(mutation))
        relative_position = position - start_idx
        assert from_AA == focus_seq[relative_position], \
            f"Invalid from_AA or mutant position: {str(mutation)} from_AA {str(str(from_AA))} " \
            f"relative pos: {str(relative_position)} focus_seq: {str(focus_seq)}"
        assert to_AA in AA_vocab, f"Mutant to_AA is invalid: {str(mutation)}"
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)


def split_kmers2(seqs: List[str], k: int = 3) -> List[List[str]]:
    return [[seq[i:i + k] for i in range(len(seq) - k + 1)] for seq in seqs]


def edit_distance(seq1: str, seq2: str) -> int:
    return levenshtein(seq1, seq2)


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
