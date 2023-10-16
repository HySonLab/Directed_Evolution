import itertools
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
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
                 conditional_task: bool = True,
                 remove_duplications: bool = False,
                 k: int = 3,
                 population_ratio_per_mask: List[float] = None,
                 use_gaussian: bool = True,
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
        self.conditional_task = conditional_task
        self.rm_dups = remove_duplications
        self.k = k
        self.use_gaussian = use_gaussian
        self.scoring_mirror = scoring_mirror
        self.batch_size_inference = batch_size_inference
        self.num_propose_mutation_per_variant = num_propose_mutation_per_variant
        self.num_workers = num_workers
        self.verbose = verbose

        self.population_ratio_per_mask = population_ratio_per_mask
        if population_ratio_per_mask is None:
            self.population_ratio_per_mask = [1 / len(maskers) for _ in range(len(maskers))]

        # Logging and caching variables
        self.mutation_logger = None
        self.prev_fitness = None
        self.prev_mutants = None
        self.prev_variants = None
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

        # <eos> token position
        eos_id = self.mutation_tokenizer.eos_token_id

        # run mutation model
        masked_inputs = self.mutation_model.tokenize(masked_variants)
        masked_outputs = self.mutation_model(masked_inputs)
        logits = masked_outputs.logits
        state = masked_outputs.hidden_states[-1]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [N, seq_len, vocab_size]
        # actual seq_len are similar => hard fix to prevent <eos> prediction at the end of seq.
        log_probs[:, -2, eos_id] = -1e6
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
        mutants = self.logger2mutants(len(mutated_seqs))

        return mutated_seqs, mutants, state

    def predict_fitness_unconditional(self,
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
        mutation_df = pd.DataFrame({"mutated_sequence": mutated_seqs, "mutant": mutants})
        log_fitness_df = self.fitness_predictor.score_mutants(mutation_df,
                                                              wt_seq,
                                                              self.scoring_mirror,
                                                              self.batch_size_inference,
                                                              self.num_workers)
        log_fitness_df = log_fitness_df if self.prev_fitness is None \
            else pd.concat([log_fitness_df, self.prev_fitness], ignore_index=True)
        top_fitness = log_fitness_df.nlargest(n=self.population,
                                              columns="avg_score",
                                              keep="first")
        self.prev_fitness = top_fitness

        # make len(top_fitness) == self.population
        if len(top_fitness) < self.population:
            n = self.population - len(top_fitness) + 1
            top_fitness = pd.concat([top_fitness.iloc[[0]]] * n + [top_fitness.iloc[1:]],
                                    ignore_index=True)

        # update self.mutation_logger according to saved mutant
        self.mutation_logger = self.mutants2logger(top_fitness.mutant.tolist())
        top_variants = top_fitness.mutated_sequence.tolist()
        top_fitness_score = top_fitness.avg_score.tolist()
        return top_variants, top_fitness_score

    def predict_fitness_conditional(self,
                                    inputs: Union[str, torch.Tensor],
                                    wt_fitness: float,
                                    mutated_seqs: List[str],
                                    mutants: List[str]) -> Union[List[str], List[float]]:
        """Third step of Directed Evolution
        Args:
            inputs (str | torch.Tensor): wild-type sequence or sequence representation shape of
                (batch, sequence_len, dim).
            wt_fitness (float): wild-type sequence's fitness.
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.

        Returns:
            new_variants (List[str]): List of mutated sequences sorted by fitness score.
        """
        if self.use_gaussian:
            assert isinstance(inputs, str)
            raise NotImplementedError
        else:
            assert isinstance(inputs, torch.Tensor)
            # (batch, 1)
            fitness = self.fitness_predictor.infer_fitness(inputs).detach()
            fitness = torch.concat([fitness, self.prev_fitness], dim=0)
            mutants = mutants + self.prev_mutants
            mutated_seqs = mutated_seqs + self.prev_variants

            # Get topk fitness score
            k = self.population if len(mutants) >= self.population else len(mutants)
            topk_fitness, topk_indices = torch.topk(fitness, k, dim=0)
            top_fitness_score = topk_fitness.squeeze(1).numpy().tolist()
            top_indices = topk_indices.squeeze(1).numpy().tolist()

            # Fill pool to fit pool size (if needed)
            n = 0
            if len(top_fitness_score) < self.population:
                n = self.population - len(top_fitness_score)
                top_fitness_score = [top_fitness_score[0] for _ in range(n)] + top_fitness_score
                top_indices = [top_indices[0] for _ in range(n)] + top_indices

            # Get top variants
            retriever = itemgetter(*top_indices)
            top_variants = list(retriever(mutated_seqs))
            top_mutants = list(retriever(mutants))

            # update self.mutation_logger according to saved mutant
            self.mutation_logger = self.mutants2logger(top_mutants)
            self.prev_fitness = topk_fitness
            self.prev_mutants = top_mutants[n:]
            self.prev_variants = top_variants[n:]

        return top_variants, top_fitness_score

    @timer
    def predict_fitness(self,
                        inputs: Union[str, torch.Tensor],
                        wt_fitness: float,
                        mutated_seqs: List[str],
                        mutants: List[str]) -> Union[List[str], List[float]]:
        """Third step of Directed Evolution
        Args:
            inputs (str | torch.Tensor): wild-type sequence or sequence representation shape of
                (batch, sequence_len, dim).
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.

        Returns:
            new_variants (List[str]): List of mutated sequences sorted by fitness score.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== FITNESS PREDICTION ======")

        if self.conditional_task:
            top_variants, top_fitness = self.predict_fitness_conditional(
                inputs, wt_fitness, mutated_seqs, mutants
            )
        else:
            top_variants, top_fitness = self.predict_fitness_unconditional(
                inputs, mutated_seqs, mutants
            )
        return top_variants, top_fitness

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

    def __call__(self, wt_seq: str, wt_fitness: float = 0.):
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
        variants = [wt_seq for _ in range(self.population)]
        self.mutation_logger = [{} for _ in range(self.population)]
        self.prev_fitness = torch.tensor([[wt_fitness]], dtype=torch.float32)
        self.prev_mutants = [""]
        self.prev_variants = [wt_seq]

        for step in range(self.n_steps):
            # ============================ #
            # ====== PRE-PROCESSING ====== #
            # ============================ #
            if self.verbose:
                now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print(f"\n{now}: ====== Step {step + 1} ======")

            if step != 0:
                variants = list(itertools.chain.from_iterable(
                    list(deepcopy(i) for _ in range(self.num_propose_mutation_per_variant))
                    for i in variants
                ))
                self.mutation_logger = list(itertools.chain.from_iterable(
                    list(deepcopy(i) for _ in range(self.num_propose_mutation_per_variant))
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
            mutated_seqs, mutants, enc_out = self.mutate_masked_sequences(wt_seq,
                                                                          masked_variants,
                                                                          masked_positions)

            # Remove duplications if needed
            mutated_seqs, mutants, enc_out = self.remove_dups(enc_out, mutated_seqs, mutants)

            # ================================ #
            # ====== FITNESS PREDICTION ====== #
            # ================================ #
            inputs = enc_out if self.conditional_task and not self.use_gaussian else wt_seq
            variants, score = self.predict_fitness(inputs, wt_fitness, mutated_seqs, mutants)

        return (self.prev_mutants, self.prev_fitness)

        # Return best variant
        best_score = score[0]
        mutant = ''
        for k, v in self.mutation_logger[0].items():
            mutant += v[0] + k + v[1] + ":"

        return (mutant[:-1], best_score)

    def remove_dups(self, enc_out, mutated_seqs, mutants):
        candidate_array = np.array(mutated_seqs)
        unique_cand, indices = np.unique(candidate_array, return_index=True)
        unique_mutated_seqs = unique_cand.tolist()
        unique_indices = indices.tolist()

        # Retrieve unique elements based on indices
        unique_enc_out = enc_out[unique_indices]
        retriever = itemgetter(*unique_indices)
        unique_mutants = list(retriever(mutants))
        self.mutation_logger = list(retriever(self.mutation_logger))

        return unique_mutated_seqs, unique_mutants, unique_enc_out

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
            if len(mutant) == 0:
                continue
            for m in mutant.split(":"):
                before, pos, after = m[0], m[1:-1], m[-1]
                logger[idx][pos] = [before, after]
        return logger
