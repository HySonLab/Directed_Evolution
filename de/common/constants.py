from itertools import product


CANONICAL_ALPHABET = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]


def all_possible_kmers(k: int):
    kmers = [''.join(comb) for comb in product(CANONICAL_ALPHABET, repeat=k)]
    return kmers
