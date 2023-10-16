import random
from typing import List
from .base import BaseMasker


class RandomMasker2(BaseMasker):
    def __init__(self, k: int = 1, max_subs: int = 5, mask_token: str = "<mask>"):
        self.k = k
        self.mask_token = mask_token
        self.max_subs = max_subs

    def mask_random_pos(self, seq: str):
        """Mask random positions in the protein sequence

        Args:
            seq (List[str]): Protein sequence.

        Returns:
            seq (str): Masked protein sequence.
            pos_to_mutate (List[int]): Masked positions.
        """
        if self.k > 1:
            assert self.max_subs == 1, "Only substitute 1 k-mer at a time for k > 1."

        lseq = list(seq)
        min_pos = 0
        max_pos = len(lseq) - self.k + 1

        candidate_masked_pos = list(range(min_pos, max_pos))
        random.shuffle(candidate_masked_pos)
        pos_to_mutate = candidate_masked_pos[:self.max_subs]

        for i in range(self.max_subs):
            pos = pos_to_mutate[i]
            lseq[pos:pos + self.k] = [self.mask_token] * self.k

        if self.k == 1:
            return ''.join(lseq), pos_to_mutate
        else:
            return ''.join(lseq), list(range(pos_to_mutate[0], pos_to_mutate[0] + self.k))

    def run(self,
            population: List[str],
            indices: List[int] = None):
        masked_population = []
        masked_positions = []
        for seq in population:
            new_seq, masked_pos = self.mask_random_pos(seq)
            masked_population.append(new_seq)
            masked_positions.append(masked_pos)
        return masked_population, masked_positions
