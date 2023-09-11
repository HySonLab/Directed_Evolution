import math
import random
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Tuple, List
from .base import BaseMasker


class ImportanceMasker(BaseMasker):
    def __init__(self,
                 use_full_dataset: bool = False,
                 full_dataset: List[List[str]] = None,
                 mask_token: str = "<mask>",
                 max_subs: int = 5,
                 low_importance_mask: bool = True):
        # TODO: mask by assigning weight by the importance?
        self.use_full_dataset = use_full_dataset
        self.mask_token = mask_token
        self.max_subs = max_subs
        self.low_importance_mask = low_importance_mask
        # calculate the importance
        self.importances = None
        if use_full_dataset:
            if full_dataset is None:
                raise ValueError("`full_dataset` must be specified as use_full_dataset = True.")
            self.importances = self._measure_importance(full_dataset)
        # cache importance of kmer (as we do not alter every kmer)
        self.cache = None
        # TF-IDF does not filter out stand-alone amino acid.
        self.tfidf = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
        self.actual_vocabs = None

    def _measure_importance(self, sequences: List[List[str]]):
        """Inspired by paper
        `A Cheaper and Better Diffusion Language Model with Soft-Masked Noise`
        """
        # TODO: vectorize this function (now using for loops)
        # TODO: handle case when multiple case mask a single index
        merge_seqs = [' '.join(seq) for seq in sequences]
        # Run TF-IDF
        tfidfs = self.tfidf.fit_transform(merge_seqs)
        self.actual_vocabs = {
            name: idx for idx, name in enumerate(self.tfidf.get_feature_names_out())
        }

        # Get entropy
        kmer2entropy = self._get_entropy_of_unique_tokens(sequences)

        # Measure importance
        importances = []
        for seq_idx, seq in enumerate(sequences):
            kmer2imp = dict()
            setseq = list(set(seq))
            seq_tfidf = tfidfs[seq_idx].sum()
            seq_entropy = 0
            seq_tfidfs = []
            for kmer in setseq:
                # Temporary
                try:
                    kmer_idx = self.actual_vocabs[kmer]
                except KeyError:
                    self.actual_vocabs[kmer] = len(self.actual_vocabs)
                    kmer_idx = self.actual_vocabs[kmer]

                tfidf = tfidfs[seq_idx, kmer_idx]
                seq_tfidfs.append(tfidf)
                seq_entropy += kmer2entropy[kmer]

            for kmer, tfidf in zip(setseq, seq_tfidfs):
                kmer2imp[kmer] = tfidf / seq_tfidf + kmer2entropy[kmer] / seq_entropy

            importances.append(kmer2imp)

        return importances

    def _get_entropy_of_unique_tokens(self, seqs: List[List[str]]):
        bag_of_toks = list(itertools.chain.from_iterable(seqs))
        total_tokens = len(bag_of_toks)
        count = {}
        for tok in bag_of_toks:
            if tok in count:
                count[tok] += 1
            else:
                count[tok] = 1

        entropy = {}
        for k, v in count.items():
            prob = v / total_tokens
            entropy[k] = -1.0 * prob * math.log(prob)

        return entropy

    def mask_sequence(self,
                      seq: List[str],
                      kmer2imp: Dict,
                      num_subs: int):
        """Mask sequence based on kmer's importance.
        Default is to mask kmers with low importances.

        Args:
            seq (List[str]): Protein sequence.
            kmer2imp (Dict): A dictionary map kmer with its importance in the sequence.
            num_subs (int): No. subsitution be made

        Returns:
            seq (List[str]): Masked protein sequence.
            pos_to_mutate (List[int]): Masked positions.
        """
        if self.low_importance_mask:
            sorted_kmers_by_imps = sorted(kmer2imp.items(), key=lambda x: x[1])
        else:
            sorted_kmers_by_imps = sorted(kmer2imp.items(), key=lambda x: x[1], reverse=True)
        sorted_kmers_by_imps = dict(sorted_kmers_by_imps)

        positions = []
        curr_idx, start_pos = 0, 0
        for _ in range(num_subs):
            try:
                pos = seq.index(list(sorted_kmers_by_imps.keys())[curr_idx], start_pos)
                start_pos = pos
            except ValueError:
                curr_idx += 1
                start_pos = 0
                pos = seq.index(list(sorted_kmers_by_imps.keys())[curr_idx], start_pos)

            seq[pos] = self.mask_token
            positions.append(pos)

        return seq, positions

    def run(self,
            population: List[List[str]],
            indices: List[int] = None,
            edit_range: Tuple[int, int] = None):

        if not self.use_full_dataset:
            importances = self._measure_importance(population)
        else:
            assert indices is not None, "`indices` must be defined."
            importances = self.importances[indices]

        masked_population = []
        masked_positions = []
        for kmer2imp, seq in zip(importances, population):
            num_edits = random.randint(1, self.max_subs)
            new_seq, masked_pos = self.mask_sequence(seq, kmer2imp, num_edits)
            masked_population.append(new_seq)
            masked_positions.append(masked_pos)
        return masked_population, masked_positions
