import random
from typing import Tuple, List
from .base import BaseMasker


class RandomMasker(BaseMasker):
    def __init__(self, mask_token: str = "<mask>", max_subs: int = 5):
        self.mask_token = mask_token
        self.max_subs = max_subs

    def mask_random_pos(self,
                        seq: List[str],
                        num_subs: int,
                        edit_range: Tuple[int, int] = None):
        """Mask random positions in the protein sequence

        Args:
            seq (List[str]): Protein sequence.
            num_subs (int): No. subsitution be made
            edit_range (Tuple[int, int]): range that edit be made

        Returns:
            seq (List[str]): Masked protein sequence.
            pos_to_mutate (List[int]): Masked positions.
        """
        if edit_range is None:
            edit_range = (1, len(seq) - 2)  # BUG: at 0 (len-1), model will replace <cls> (<eos>).
        else:
            assert edit_range[0] < 0, \
                "The lower bound of `edit_range` must not negative."
            assert edit_range[1] >= len(seq), \
                "The upper bound of `edit_range` must smaller than length."
        min_pos = edit_range[0]
        max_pos = edit_range[1]

        candidate_masked_pos = list(range(min_pos, max_pos + 1))
        random.shuffle(candidate_masked_pos)
        pos_to_mutate = candidate_masked_pos[:num_subs]

        for i in range(num_subs):
            pos = pos_to_mutate[i]
            seq[pos] = self.mask_token
        return seq, pos_to_mutate

    def run(self,
            population: List[List[str]],
            indices: List[int] = None,
            edit_range: Tuple[int, int] = None):
        masked_population = []
        masked_positions = []
        for seq in population:
            num_edits = random.randint(1, self.max_subs)
            new_seq, masked_pos = self.mask_random_pos(seq, num_edits, edit_range)
            masked_population.append(new_seq)
            masked_positions.append(masked_pos)
        return masked_population, masked_positions
