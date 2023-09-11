from typing import List, Dict


def read_fasta(
    filepath: str,
    do_filter: bool = True,
    max_seq_length: int = 1024,
    excluded_residues: List[str] = ['X', 'Z', 'B', 'U', 'O']
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
                    sequences[sequence_id] = sequence
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line.strip()
        if sequence_id:
            sequences[sequence_id] = sequence

    if do_filter:
        sequences = filter_seqs(sequences, max_seq_length, excluded_residues)

    return sequences


def filter_seqs(
    sequences: List[str],
    max_length: int = 1024,
    excluded_residues: List[str] = ['X', 'Z', 'B', 'U', 'O']
) -> List[str]:
    new_seqs = {}
    for id, seq in sequences.items():
        if max_length > 0 and len(seq) > max_length:
            continue
        if any(residue in seq for residue in excluded_residues):
            continue
        new_seqs[id] = seq
    return new_seqs
