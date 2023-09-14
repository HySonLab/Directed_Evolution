import itertools
import numpy as np
import os
import pandas as pd
import torch
from datetime import datetime
from operator import itemgetter
from typing import List, Tuple
from de.common.utils import split_kmers, timer
from de.samplers.maskers import BaseMasker
from transformers import PreTrainedTokenizer


class DiscreteDirectedEvolution:
    def __init__(self,
                 n_steps: int,
                 population: int,
                 maskers: List[BaseMasker],
                 mutation_model: torch.nn.Module,
                 mutation_tokenizer: PreTrainedTokenizer,
                 fitness_predictor: torch.nn.Module,
                 k: int = 3,
                 population_ratio_per_mask: List[float] = None,
                 scoring_mirror: bool = False,
                 batch_size_inference: int = 5,
                 num_propose_mutation_per_variant: int = 5,
                 edit_range: Tuple[int, int] = None,
                 mutation_ckpt_path: str = None,
                 verbose: bool = False,
                 num_workers: int = 16,
                 device: str = "cpu"):
        """Main class for Discrete-space Directed Evolution

        Args:
            n_steps (int): No. steps to run directed evolution
            population (int): No. population per run
            verbose (bool): Whether to print output
        """
        self.n_steps = n_steps
        self.population = population
        self.maskers = maskers
        self.mutation_model = mutation_model
        self.mutation_tokenizer = mutation_tokenizer
        self.fitness_predictor = fitness_predictor
        self.k = k
        self.scoring_mirror = scoring_mirror
        self.batch_size_inference = batch_size_inference
        self.num_propose_mutation_per_variant = num_propose_mutation_per_variant
        self.edit_range = edit_range
        self.num_workers = num_workers
        self.verbose = verbose
        self.device = device

        self.population_ratio_per_mask = population_ratio_per_mask
        if population_ratio_per_mask is None:
            self.population_ratio_per_mask = [1 / len(maskers) for _ in range(len(maskers))]

        self.mutation_logger = None

        if mutation_ckpt_path is not None:
            if not os.path.exists(mutation_tokenizer):
                raise ValueError(f"{mutation_ckpt_path} is not exists.")
            self.load_mutation_checkpoint(mutation_ckpt_path)
        self.mutation_model.to(self.device)
        self.mutation_model.eval()

        # Checks
        if self.n_steps < 1:
            raise ValueError("`n_steps` must be >= 1")
        if self.k < 1:
            raise ValueError("`k` must be >= 1")

        if population_ratio_per_mask is None:
            population_ratio_per_mask = [1.0 / len(maskers)] * len(maskers)

    def load_mutation_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path)
        self.mutation_model.load_state_dict(checkpoint["model_state_dict"])

    @timer
    def mask_sequences(
        self,
        variants: List[str],
        ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """First step in Directed Evolution
        Args:
            variants (List[str]): List of sequences to be masked.
            ids (List[str]): Corresponding indices of `variants` w.r.t original list.

        Returns:
            masked_variants (List[List[str]]): Masked sequences (each has been splitted into k-mers)
            masked_poses (List[int]): Masked positions.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== MASK VARIANTS ======")
            print(f"{now}: Start masking {self.population} variants.")

        masked_variants, masked_poses = [], []
        begin_idx = 0
        for population_ratio, masker in zip(self.population_ratio_per_mask, self.maskers):
            sub_population = int(self.population * population_ratio)
            sub_variants = variants[begin_idx:begin_idx + sub_population]
            sub_ids = ids[begin_idx:begin_idx + sub_population]
            begin_idx += sub_population
            # split into list of amino acids/k-mers
            sub_variants = split_kmers(sub_variants, k=self.k)

            masked_vars, masked_pos = masker.run(sub_variants,
                                                 sub_ids,
                                                 self.edit_range)
            masked_variants.extend(masked_vars)
            masked_poses.extend(masked_pos)

        return masked_variants, masked_poses

    @timer
    def mutate_masked_sequences(
        self,
        wt_seq: str,
        masked_variants: List[List[str]],
        masked_poses: List[int]
    ) -> Tuple[List[str], List[str]]:
        """Second step of Directed Evolution
        Args:
            wt_seq (str): wild-type sequence.
            masked_variants (List[List[str]]): Masked sequences (each has been splitted into k-mers)
            masked_poses (List[int]): Masked positions.

        Returns:
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== MUTATE MASKED POSITION ======")

        # process input
        masked_data = [''.join(var) for var in masked_variants]

        # run mutation model
        masked_inputs = self.mutation_model.tokenize(masked_data)
        logits = self.mutation_model(masked_inputs).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [N, seq_len, vocab_size]
        predicted_seqs = torch.argmax(log_probs, dim=-1)  # [N, seq_len] (N ~ population)
        predicted_seqs = torch.chunk(predicted_seqs, self.population)  # List of [1, seq_len]

        # replace predicted token with <masked> token
        mutated_seqs = []
        mutants = []
        # TODO: Think a way to vectorize this!!!!
        for idx in range(self.population):
            tposes = torch.tensor(masked_poses[idx])
            tposes = tposes.unsqueeze(0) if tposes.ndim == 0 else tposes
            tokens = torch.index_select(
                predicted_seqs[idx].squeeze(), dim=0, index=tposes).numpy().tolist()

            mutated_seq = masked_variants[idx]
            mutant = ''
            for tok, pos in zip(tokens, masked_poses[idx]):

                # print(f"Before:  mutated_seq[{pos}] = {wt_seq[pos]}")
                mutated_seq[pos] = self.mutation_tokenizer.decode(tok)
                # print(f"After :  mutated_seq[{pos}] = {mutated_seq[pos]}\n")

                self.mutation_logger[idx][str(pos + 1)] = [wt_seq[pos], mutated_seq[pos]]

            mutated_seq = ''.join(mutated_seq)
            mutated_seqs.append(mutated_seq)
            for k, v in self.mutation_logger[idx].items():
                mutant += v[0] + k + v[1] + ":"
            mutants.append(mutant[:-1])

        return mutated_seqs, mutants

    @timer
    def predict_fitness(self,
                        wt_seq: str,
                        mutated_seqs: List[str],
                        mutants: List[str]) -> List[str]:
        """Third step of Directed Evolution
        Args:
            wt_seq (str): wild-type sequence.
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.

        Returns:
            new_variants (List[str]): List of mutated sequences sorted by fitness score.
        """
        mutation_df = pd.DataFrame({"mutated_sequence": mutated_seqs, "mutant": mutants})
        log_fitness_df = self.fitness_predictor.score_mutants(mutation_df,
                                                              wt_seq,
                                                              self.scoring_mirror,
                                                              self.batch_size_inference,
                                                              self.num_workers)
        top_log_fitness = log_fitness_df.nlargest(n=self.population,
                                                  columns="avg_score",
                                                  keep="first")
        # Re-arrange mutation_logger
        sorted_ids = top_log_fitness.index.tolist()
        retriever = itemgetter(*sorted_ids)
        self.mutation_logger = list(retriever(self.mutation_logger))

        return top_log_fitness

    def __call__(self, wt_seq: str):
        """Run the discrete-space directed evolution

        Args:
            wt_seq (str): wild-type protein sequence

        Returns:
            variants (List[str]): list of protein sequences
            scores (torch.Tensor): scores for the variants
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"{now}: Wild-type sequence: {wt_seq}")

        # Initialize
        variants = [wt_seq] * self.population
        self.mutation_logger = [{} for _ in range(self.population)]

        for step in range(self.n_steps):
            # TODO: beam search-like mutation (each variant has more than 1 proposed mutation)
            # ============================ #
            # ====== PRE-PROCESSING ====== #
            # ============================ #
            if step != 0:
                variants = list(itertools.chain.from_iterable(
                    itertools.repeat(i, self.num_propose_mutation_per_variant)
                    for i in variants
                ))
                self.mutation_logger = list(itertools.chain.from_iterable(
                    itertools.repeat(i, self.num_propose_mutation_per_variant)
                    for i in self.mutation_logger
                ))
            shuffled_ids = np.random.permutation(len(variants)).tolist()
            retriever = itemgetter(*shuffled_ids)
            shuffled_variants = list(retriever(variants))
            if step != 0:
                self.mutation_logger = list(retriever(self.mutation_logger))
            del retriever

            # =========================== #
            # ====== MASK VARIANTS ====== #
            # =========================== #
            masked_variants, masked_poses = self.mask_sequences(shuffled_variants, shuffled_ids)

            # ==================================== #
            # ====== MUTATE MASKED POSITION ====== #
            # ==================================== #
            mutated_seqs, mutants = self.mutate_masked_sequences(wt_seq,
                                                                 masked_variants,
                                                                 masked_poses)

            # ================================ #
            # ====== FITNESS PREDICTION ====== #
            # ================================ #
            top_fitness = self.predict_fitness(wt_seq, mutated_seqs, mutants)
            variants = top_fitness.mutated_sequence.tolist()

        # Return best variant
        best_fitness = top_fitness.nlargest(1, columns="avg_score")
        # best_variant = str(best_fitness.mutated_sequence.iloc[0])
        best_score = float(best_fitness.avg_score.iloc[0])
        mutant = ''
        for k, v in self.mutation_logger[0].items():
            mutant += v[0] + k + v[1] + ":"

        return (mutant[:-1], best_score)
