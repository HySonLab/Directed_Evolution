from abc import ABC, abstractmethod
from typing import List


class BaseMasker(ABC):
    """Base class for maskers."""
    @abstractmethod
    def run(self,
            population: List[str],
            indices: List[int] = None):
        """
        Args:
            population (List[str]): List of sequences to be masked
            indices (List[int]): List of indices of each sequence in original population.
        Returns:
            masked_population (List[str]): List of masked sequence
            masked_poses (List[List[int]]): List of masked positions for each sequence.
        """
        raise NotImplementedError
