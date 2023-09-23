import re
from typing import List, Dict
from .constants import CANONICAL_ALPHABET


def read_fasta(
    filepath: str,
    do_filter: bool = True,
    max_seq_length: int = 1024,
    accepted_residues: List[str] = CANONICAL_ALPHABET,
) -> Dict[str, str]:
    """ Read a fasta file

    Args:
        filepath (str): path to fasta file

    Returns:
        sequences (dict): map multiple sequence ids to corresponding sequences."""
    sequences = {}
    with open(filepath, 'r') as file:
        sequence_id = None
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id:
                    sequences[sequence_id] = sequence.upper()
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line.strip()
        if sequence_id:
            sequences[sequence_id] = sequence.upper()

    if do_filter:
        sequences = filter_seqs(sequences, max_seq_length, accepted_residues)

    return sequences


def filter_seqs(
    sequences: List[str],
    max_length: int = 1024,
    accepted_residues: List[str] = CANONICAL_ALPHABET
) -> List[str]:
    valid_residues = "".join(accepted_residues)

    def contains_invalid_chars(input):
        pattern = f"[^{re.escape(valid_residues)}]"
        return bool(re.search(pattern, input))

    new_seqs = {}
    for id, seq in sequences.items():
        if max_length > 0 and len(seq) > max_length:
            continue
        if contains_invalid_chars(seq):
            continue
        new_seqs[id] = seq
    return new_seqs
