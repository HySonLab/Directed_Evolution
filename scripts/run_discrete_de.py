import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.multiprocessing as mp
import traceback
from pickle import dump, load
from typing import List, Union, Tuple
from transformers import PreTrainedTokenizerFast
from de.common.io_utils import read_fasta
from de.common.utils import set_seed, enable_full_deterministic
from de.directed_evolution import DiscreteDirectedEvolution2
from de.samplers.maskers import RandomMasker2, ImportanceMasker2
from de.samplers.models.esm import ESM2
from de.predictors.tranception.model import TranceptionLMHeadModel
from de.predictors.gaussian_process import GaussianProcess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file",
                        nargs="+",
                        type=str,
                        default=["../data/uniprot_sprot.fasta"],
                        help="Path to fasta file.")
    parser.add_argument("--csv_file",
                        nargs="+",
                        type=str,
                        default=["../data/peptides/apd.csv"],
                        help="Path to csv file.")
    parser.add_argument("--tokenizer_file",
                        type=str,
                        default="../de/predictors/tranception/utils/tokenizers/Basic_tokenizer",
                        help="Tokenizer file to initialize Tranception's tokenizer.")
    parser.add_argument("--max_sequence_length",
                        type=int,
                        default=-1,
                        help="Maximum sequence length accepted.")
    parser.add_argument("--n_steps",
                        type=int,
                        default=30,
                        help="No. steps to run directed evolution.")
    parser.add_argument("--population",
                        type=int,
                        default=20,
                        help="No. population per step.")
    parser.add_argument("--num_proposes_per_var",
                        type=int,
                        default=10,
                        help="Number of proposed mutations for each variant in the pool.")
    parser.add_argument("--k",
                        type=int,
                        default=1,
                        help="Split sequence into multiple tokens with length `k`.")
    parser.add_argument("--tranception_type",
                        type=str,
                        choices=["Small", "Medium", "Large"],
                        default="Small",
                        help="Choose Tranception model size.")
    parser.add_argument("--pretrained_gaussian",
                        type=str,
                        help="Path to pretrained Gaussian Process model (saving or loading).")
    parser.add_argument("--population_ratio_per_mask",
                        nargs="+",
                        type=float,
                        help="Population ratio to run per masker.")
    parser.add_argument("--pretrained_mutation_name",
                        type=str,
                        default="facebook/esm2_t12_35M_UR50D",
                        help="Pretrained model name or path for mutation checkpoint.")
    parser.add_argument("--num_masked_tokens",
                        type=int,
                        default=1,
                        help="No. masked tokens to predict.")
    parser.add_argument("--mask_high_importance",
                        action="store_true",
                        help="Whether to mask high-importance token in the sequence.")
    parser.add_argument("--scoring_mirror",
                        action="store_true",
                        help="Whether to measure fitness from right to left.")
    parser.add_argument("--fitness_optim",
                        type=str,
                        choices=["pareto", "ranking"],
                        default="ranking",
                        help="Algorithm to choose top best fitness score.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Whether to display output.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=10,
                        help="Batch size for inference.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed.")
    parser.add_argument("--set_seed_only",
                        action="store_true",
                        help="Whether to enable full determinism or set random seed only.")
    parser.add_argument("--num_processes",
                        type=int,
                        default=mp.cpu_count() // 2,
                        help="No. cpus used for multi-processing.")
    parser.add_argument("--result_dir",
                        type=str,
                        default="/home/thanhtvt1/workspace/Directed_Evolution/exps/results",
                        help="Directory to save result csv file.")
    parser.add_argument("--save_name",
                        type=str,
                        help="Filename of the result csv file.")
    parser.add_argument("--use_tranception",
                        action="store_true",
                        help="Whether to use Tranception for fitness prediction.")
    parser.add_argument("--max_mismatch",
                        type=int,
                        default=0,
                        help="Maximum number of mismatches to consider similar.")
    args = parser.parse_args()
    return args


def get_sequence_from_fasta(fasta_files: List[str], max_seq_len: int) -> List[str]:
    seqs = []
    for filepath in fasta_files:
        fasta_seqs = read_fasta(filepath, max_seq_length=max_seq_len)
        fasta_seqs = list(fasta_seqs.values())
        seqs.extend(fasta_seqs)
    return seqs


def extract_from_csv(csv_files: List[str]) -> Tuple[List[str], np.ndarray]:
    seqs = []
    targets = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        insta_idx = df.instability_index.tolist()
        boman_idx = df.Boman_index.tolist()
        seqs.extend(df.sequence.tolist())
        targets.extend([[y1, y2] for y1, y2 in zip(insta_idx, boman_idx)])
    targets = np.array(targets, dtype=np.float32)
    return seqs, targets


def initialize_mutation_model(args, device: torch.device):
    model = ESM2(pretrained_model_name_or_path=args.pretrained_mutation_name,
                 device=device)
    tokenizer = model.tokenizer
    model = model.to(device)
    return model, tokenizer


def initialize_maskers(args):
    imp_masker = ImportanceMasker2(args.k,
                                   max_subs=args.num_masked_tokens,
                                   low_importance_mask=not args.mask_high_importance)
    rand_masker = RandomMasker2(args.k, max_subs=args.num_masked_tokens)

    return [imp_masker, rand_masker]


def intialize_fitness_predictor(args, device: Union[str, torch.device]):
    if args.use_tranception:
        model_name = f"PascalNotin/Tranception_{args.model_size}"
        model = TranceptionLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=model_name
        )
        model.config.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=args.tokenizer_file,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]"
        )
        model = model.to(device)
    else:
        if args.pretrained_gaussian and os.path.isfile(args.pretrained_gaussian):
            with open(args.pretrained_gaussian, "rb") as f:
                model = load(f)
        else:
            model = GaussianProcess(k=args.k,
                                    alphabet_size=20,
                                    max_num_mismatch=args.max_mismatch)

    return model


def save_results(evos: List[Tuple[str, float]], output_path: str):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    mutants, score = zip(*evos)
    df = pd.DataFrame.from_dict({"mutants": list(mutants), "score": list(score)})
    df.to_csv(output_path, index=False)


def main(args):
    # Set up multiprocessing
    # torch.set_num_threads(1)
    # mp.set_start_method("spawn", force=True)

    # Get sequences
    if not args.use_tranception:
        sequences, targets = extract_from_csv(args.csv_file)
    else:
        sequences = get_sequence_from_fasta(args.fasta_file, args.max_sequence_length)

    # Init env stuffs
    set_seed(args.seed) if args.set_seed_only else enable_full_deterministic(args.seed)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = 'true'
    device = torch.device("cpu")

    # Init models
    mutation_model, mutation_tokenizer = initialize_mutation_model(args, device)
    fitness_predictor = intialize_fitness_predictor(args, device)

    # Init masker
    maskers = initialize_maskers(args)

    # pre-fit Gaussian Process
    if not args.use_tranception and not os.path.isfile(args.pretrained_gaussian):
        fitness_predictor.fit(sequences, targets)
        with open(args.pretrained_gaussian, "wb") as f:
            dump(fitness_predictor, f)
        # fitness_predictor.save_model(args.pretrained_gaussian)

    # Init procedure
    direct_evo = DiscreteDirectedEvolution2(
        n_steps=args.n_steps,
        population=args.population,
        maskers=maskers,
        mutation_model=mutation_model,
        mutation_tokenizer=mutation_tokenizer,
        fitness_predictor=fitness_predictor,
        fitness_optim=args.fitness_optim,
        k=args.k,
        population_ratio_per_mask=args.population_ratio_per_mask,
        use_tranception=args.use_tranception,
        scoring_mirror=args.scoring_mirror,
        batch_size_inference=args.batch_size,
        num_propose_mutation_per_variant=args.num_proposes_per_var,
        verbose=args.verbose,
    )

    evos = []
    for seq in sequences:
        evo = direct_evo(seq)
        evos.append(evo)

    # try:
    #     pool = mp.Pool(processes=args.num_processes)
    #     evos = pool.map(direct_evo, sequences[:100])
    # except Exception as e:
    #     print(traceback.format_exc())
    #     print("Main Pool Error:", e)
    # except KeyboardInterrupt:
    #     exit()
    # finally:
    #     pool.terminate()
    #     pool.join()

    filename = args.save_name or "results_" + os.path.basename(args.csv_file[0])
    filepath = os.path.join(args.result_dir, filename)
    save_results(evos, filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
