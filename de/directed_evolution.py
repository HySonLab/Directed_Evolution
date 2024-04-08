import itertools
import logging
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime
from operator import itemgetter
from typing import List, Tuple, Union
from transformers import PreTrainedTokenizer
from de.common.utils import timer
from de.samplers.maskers import BaseMasker


class DiscreteDirectedEvolution2:
    def __init__(self,
                 n_steps: int,
                 population: int,
                 maskers: List[BaseMasker],
                 mutation_model: torch.nn.Module,
                 mutation_tokenizer: PreTrainedTokenizer,
                 fitness_predictor: Union[torch.nn.Module, object],
                 remove_duplications: bool = False,
                 k: int = 3,
                 population_ratio_per_mask: List[float] = None,
                 num_propose_mutation_per_variant: int = 5,
                 verbose: bool = False,
                 num_workers: int = 16,
                 mutation_device: Union[torch.device, str] = "cpu",
                 log_dir: str = "./logs/",
                 seed: int = 0,):
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
        self.rm_dups = remove_duplications
        self.k = k
        self.num_propose_mutation_per_variant = num_propose_mutation_per_variant
        self.num_workers = num_workers
        self.verbose = verbose
        self.mutation_device = mutation_device
        self.seed = seed
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

        filename = f"{log_dir}/log_mask={'-'.join([str(msk) for msk in self.population_ratio_per_mask])}_k={k}_beam={num_propose_mutation_per_variant}_{self.seed}.log"
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )

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

            if len(sub_variants) == 0:
                continue
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
            masked_variants (List[str]): Masked sequences (each has been splitted into k-mers).
            masked_positions (List[List[int]]): Masked positions.

        Returns:
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== MUTATE MASKED POSITION ======")

        # <eos> token position
        eos_id = self.mutation_tokenizer.eos_token_id
        masked_inputs = self.mutation_model.tokenize(masked_variants)
        # move to device
        masked_inputs.to(self.mutation_device)
        with torch.inference_mode():
            masked_outputs = self.mutation_model(masked_inputs)
            logits = masked_outputs.logits
            state = masked_outputs.hidden_states[-1]
        # return to cpu
        masked_inputs = masked_inputs.to(torch.device("cpu"))
        logits = logits.to(torch.device("cpu"))
        state = state.to(torch.device("cpu"))

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

    @timer
    def predict_fitness(self,
                        inputs: Union[str, torch.Tensor],
                        wt_fitness: float,
                        mutated_seqs: List[str],
                        mutants: List[str],
                        wt_seq: str = None) -> Union[List[str], List[float]]:
        """Third step of Directed Evolution
        Args:
            inputs (str | torch.Tensor): wild-type sequence or sequence representation shape of
                (batch, sequence_len, dim).
            mutated_seqs (List[str]): Mutated sequences
            mutants (List[str]): List of strings indicates the mutations in each sequence.

        Returns:
            top_variants (List[str]): List of mutated sequences sorted by fitness score.
            top_fitness_score (List[float]): List of fitness score sorted in descending order.
        """
        if self.verbose:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{now}: ====== FITNESS PREDICTION ======")

        inputs = inputs.to(self.mutation_device)

        # (batch, 1)
        # fitness = self.fitness_predictor.infer_fitness(inputs).detach().cpu()
        # fitness = torch.concat([fitness, self.prev_fitness], dim=0)
        fitness = torch.tensor(self.fitness_predictor.infer_fitness(mutated_seqs),
                               dtype=torch.float32)
        fitness = fitness.unsqueeze(1) if fitness.ndim == 1 else fitness
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

    def __call__(self, wt_seq: str, wt_fitness: float):
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
            inputs = enc_out
            variants, score = self.predict_fitness(
                inputs, wt_fitness, mutated_seqs, mutants, wt_seq
            )

            logging.info(f"\n-------- STEP {step} --------")
            for i, (var, mut, s) in enumerate(zip(variants, self.prev_mutants, score)):
                logging.info(f"{i}:\t{s}\t{mut}\t{var}")

        return self.prev_mutants, self.prev_fitness, variants

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
