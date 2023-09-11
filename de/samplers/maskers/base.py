from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseMasker(ABC):
    """Base class for maskers."""
    @abstractmethod
    def run(self,
            population: List[List[str]],
            indices: List[int] = None,
            edit_range: Tuple[int, int] = None):
        """
        Args:
            population (List[List[str]]): List of sequences to be masked
            indices (List[int]): List of indices of each sequence in original population.
            edit_range (Tuple[int, int]): Range to mask sequence
        Returns:
            masked_population (Union[List[str], List[List[str]]]): List of masked sequence
            masked_poses (List[int]): List of masked positions for each sequence.
        """
        raise NotImplementedError
