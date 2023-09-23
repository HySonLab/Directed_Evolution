import itertools
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from operator import itemgetter
from typing import List, Tuple, Union
from de.common.utils import timer
from de.samplers.maskers import BaseMasker
from transformers import PreTrainedTokenizer


class DiscreteDirectedEvolution2:
    def __init__(self,
                 n_steps: int,
                 population: int,
                 maskers: List[BaseMasker],
                 mutation_model: torch.nn.Module,
                 mutation_tokenizer: PreTrainedTokenizer,
                 fitness_predictor: Union[torch.nn.Module, object],
                 fitness_optim: str = "ranking",
                 k: int = 3,
                 population_ratio_per_mask: List[float] = None,
                 use_tranception: bool = True,
                 scoring_mirror: bool = False,
                 batch_size_inference: int = 5,
                 num_propose_mutation_per_variant: int = 5,
                 verbose: bool = False,
                 num_workers: int = 16):
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
        self.fitness_optim = fitness_optim
        self.k = k
        self.use_tranception = use_tranception
        self.scoring_mirror = scoring_mirror
        self.batch_size_inference = batch_size_inference
        self.num_propose_mutation_per_variant = num_propose_mutation_per_variant
        self.num_workers = num_workers
        self.verbose = verbose

        self.population_ratio_per_mask = population_ratio_per_mask
        if population_ratio_per_mask is None:
            self.population_ratio_per_mask = [1 / len(maskers) for _ in range(len(maskers))]

        self.mutation_logger = None
        # Checks
        if self.n_steps < 1:
            raise ValueError("`n_steps` must be >= 1")
        if self.k < 1:
            raise ValueError("`k` must be >= 1")

    @timer
    def mask_sequences(
        self,
        variants: List[str],
        ids: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        """First step in Directed Evolution
        Args:
            variants (List[str]): List of sequences to be masked.
            ids (List[int]): Corresponding indices of `variants` w.r.t original list.

        Returns:
            masked_variants (List[str]): Masked sequences
            masked_poses (List[List[int]]): Masked positions.
        """
        num_variant = len(variants)
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== MASK VARIANTS ======")
            print(f"{now}: Start masking {num_variant} variants.")

        masked_variants, masked_positions = [], []
        begin_idx = 0
        for population_ratio, masker in zip(self.population_ratio_per_mask, self.maskers):
            sub_population = int(num_variant * population_ratio)
            sub_variants = variants[begin_idx:begin_idx + sub_population]
            sub_ids = ids[begin_idx:begin_idx + sub_population]
            begin_idx += sub_population

            masked_vars, masked_pos = masker.run(sub_variants, sub_ids)
            masked_variants.extend(masked_vars)
            masked_positions.extend(masked_pos)

        return masked_variants, masked_positions

    @timer
    def mutate_masked_sequences(
        self,
        wt_seq: str,
        masked_variants: List[str],
        masked_positions: List[List[int]]
    ) -> Tuple[List[str], List[str]]:
        """Second step of Directed Evolution
        Args:
            wt_seq (str): wild-type sequence.
            masked_variants (List[str]): Masked sequences (each has been splitted into k-mers)
            masked_poses (List[List[int]]): Masked positions.

        Returns:
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== MUTATE MASKED POSITION ======")

        # run mutation model
        masked_inputs = self.mutation_model.tokenize(masked_variants)
        logits = self.mutation_model(masked_inputs).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [N, seq_len, vocab_size]
        predicted_toks = torch.argmax(log_probs, dim=-1)  # [N, seq_len] (N ~ num_variant)
        # get masked positions (<cls> added to the beginning)
        masked_positions_tensor = torch.tensor(masked_positions, dtype=torch.int64) + 1
        # get mutations
        mutations = torch.gather(predicted_toks, dim=1, index=masked_positions_tensor)
        mutated_toks = masked_inputs["input_ids"].scatter_(1, masked_positions_tensor, mutations)
        mutated_seqs = self.mutation_tokenizer.batch_decode(mutated_toks, skip_special_tokens=True)
        mutated_seqs = [seq.replace(" ", "") for seq in mutated_seqs]

        mutants = []
        for idx, (posis, seq) in enumerate(zip(masked_positions, mutated_seqs)):
            for i in posis:
                self.mutation_logger[idx][str(i + 1)] = [wt_seq[i], seq[i]]
        if self.use_tranception:
            mutants = self.logger2mutants(len(mutated_seqs))

        return mutated_seqs, mutants

    @timer
    def predict_fitness(self,
                        wt_seq: str,
                        mutated_seqs: List[str],
                        mutants: List[str]) -> Union[List[str], List[float]]:
        """Third step of Directed Evolution
        Args:
            wt_seq (str): wild-type sequence.
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.

        Returns:
            new_variants (List[str]): List of mutated sequences sorted by fitness score.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== FITNESS PREDICTION ======")

        if self.use_tranception:
            mutation_df = pd.DataFrame({"mutated_sequence": mutated_seqs, "mutant": mutants})
            log_fitness_df = self.fitness_predictor.score_mutants(mutation_df,
                                                                  wt_seq,
                                                                  self.scoring_mirror,
                                                                  self.batch_size_inference,
                                                                  self.num_workers)
            top_fitness = log_fitness_df.nlargest(n=self.population,
                                                  columns="avg_score",
                                                  keep="first")
            top_variants = top_fitness.mutated_sequence.tolist()
            top_fitness_score = top_fitness.avg_score.tolist()
        else:
            # Use Gaussian Process to predict score
            fitness = self.fitness_predictor.score_mutants(mutated_seqs)
            # get top k
            if self.fitness_optim == "ranking":
                top_fitness_indices = self.rank_descriptor(
                    fitness[:, 0], fitness[:, 1])[:self.population]
            elif self.fitness_optim == "pareto":
                raise NotImplementedError("Not yet supported.")

            top_fitness_score = fitness[top_fitness_indices].tolist()
            retriever = itemgetter(*top_fitness_indices)
            top_variants = list(retriever(mutated_seqs))
            # re-order logger
            self.mutation_logger = list(retriever(self.mutation_logger))

        return top_variants, top_fitness_score

    def rank_descriptor(self, insta_idx: np.ndarray, boman_idx: np.ndarray):
        """Rank samples based on descriptors.
        `insta_idx` is better when lower
        `boman_idx` is better when higher
        """
        sorted_ids_insta = np.argsort(insta_idx)
        sorted_ids_boman = np.argsort(boman_idx)[::-1]
        rank_point = np.argsort(sorted_ids_insta) + np.argsort(sorted_ids_boman)
        final_rank = np.argsort(rank_point)
        return final_rank.tolist()

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
            # ============================ #
            # ====== PRE-PROCESSING ====== #
            # ============================ #
            if self.verbose:
                now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print(f"\n{now}: ====== Step {step + 1} ======")

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
            masked_variants, masked_positions = self.mask_sequences(shuffled_variants, shuffled_ids)

            # ==================================== #
            # ====== MUTATE MASKED POSITION ====== #
            # ==================================== #
            mutated_seqs, mutants = self.mutate_masked_sequences(wt_seq,
                                                                 masked_variants,
                                                                 masked_positions)

            # ================================ #
            # ====== FITNESS PREDICTION ====== #
            # ================================ #
            variants, score = self.predict_fitness(wt_seq, mutated_seqs, mutants)

        # Return best variant
        best_score = score[0]
        mutant = ''
        for k, v in self.mutation_logger[0].items():
            mutant += v[0] + k + v[1] + ":"

        return (mutant[:-1], best_score)

    def logger2mutants(self, num2convert: int):
        mutants = []
        for i in range(num2convert):
            mutant = ''
            for k, v in self.mutation_logger[i].items():
                mutant += v[0] + k + v[1] + ":"
            mutants.append(mutant[:-1])
        return mutants

    def mutants2logger(self, mutants: List[str]):
        logger = [{} for _ in range(len(mutants))]
        for idx, mutant in enumerate(mutants):
            print(mutant)
            for m in mutant.split(":"):
                before, pos, after = m[0], m[1], m[2]
                logger[idx][str(int(pos) + 1)] = [before, after]
        return logger
