import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.multiprocessing as mp
from typing import List, Union, Tuple
from transformers import PreTrainedTokenizerFast
from de.common.utils import set_seed, enable_full_deterministic
from de.directed_evolution import DiscreteDirectedEvolution2
from de.samplers.maskers import RandomMasker2, ImportanceMasker2
from de.samplers.models.esm import ESM2
from de.predictors.tranception.model import TranceptionLMHeadModel
from de.predictors.attention.module import ESM2DecoderModule, ESM2_Attention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        type=str,
                        help="Path to data file.")
    parser.add_argument("--tokenizer_file",
                        type=str,
                        default="../de/predictors/tranception/utils/tokenizers/Basic_tokenizer",
                        help="Tokenizer file to initialize Tranception's tokenizer.")
    parser.add_argument("--wt",
                        type=str,
                        help="Amino acid sequence.")
    parser.add_argument("--wt_fitness",
                        type=float,
                        help="Wild-type sequence's fitness.")
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
    parser.add_argument("--unconditional",
                        action="store_true",
                        help="Whether to perform conditional or unconditional task.")
    parser.add_argument("--rm_dups",
                        action="store_true",
                        help="Whether to remove duplications in the proposed candidate pool.")
    parser.add_argument("--tranception_type",
                        type=str,
                        choices=["Small", "Medium", "Large"],
                        default="Small",
                        help="Choose Tranception model size.")
    parser.add_argument("--pretrained_gaussian",
                        type=str,
                        help="Path to pretrained Gaussian Process model.")
    parser.add_argument("--population_ratio_per_mask",
                        nargs="+",
                        type=float,
                        help="Population ratio to run per masker.")
    parser.add_argument("--pretrained_mutation_name",
                        type=str,
                        default="facebook/esm2_t12_35M_UR50D",
                        help="Pretrained model name or path for mutation checkpoint.")
    parser.add_argument("--dec_hidden_size",
                        type=int,
                        default=1280,
                        help="Decoder hidden size (for conditional task).")
    parser.add_argument("--predictor_ckpt_path",
                        type=str,
                        help="Path to fitness predictor checkpoints.")
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
    parser.add_argument("--use_gaussian",
                        action="store_true",
                        help="Whether to use Gaussian Process for fitness prediction.")
    parser.add_argument("--max_mismatch",
                        type=int,
                        default=0,
                        help="Maximum number of mismatches to consider similar.")
    parser.add_argument("--start_index",
                        type=int,
                        default=0,
                        help="Continue to run from `start_index` sequence in csv file.")
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
    parser.add_argument("--save_interval",
                        type=int,
                        default=-1,
                        help="Interval to save results (-1 means save when finishing).")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Devices, separated by commas.")
    args = parser.parse_args()
    return args


def extract_from_csv(csv_file: str, top_k: int = -1) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_file)
    if top_k != -1:
        df = df.nlargest(top_k, columns="fitness")
    targets = df["fitness"].to_list()
    seqs = df.sequence.tolist()
    return seqs, targets


def initialize_mutation_model(args, device: torch.device):
    model = ESM2(pretrained_model_name_or_path=args.pretrained_mutation_name)
    tokenizer = model.tokenizer
    model.to(device)
    model.eval()
    return model, tokenizer


def initialize_maskers(args):
    imp_masker = ImportanceMasker2(args.k,
                                   max_subs=args.num_masked_tokens,
                                   low_importance_mask=not args.mask_high_importance)
    rand_masker = RandomMasker2(args.k, max_subs=args.num_masked_tokens)

    return [rand_masker, imp_masker]


def intialize_fitness_predictor(args, device: Union[str, torch.device]):
    if args.unconditional:
        model_name = f"PascalNotin/Tranception_{args.tranception_type}"
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
        model.to(device)
        model.eval()
    else:
        if args.use_gaussian:
            pass
        else:
            decoder = ESM2_Attention(args.pretrained_mutation_name, hidden_dim=args.dec_hidden_size)
            model = ESM2DecoderModule.load_from_checkpoint(
                args.predictor_ckpt_path, map_location=device, net=decoder
            )
            model.eval()

    return model


def save_results(wt_seqs: List[str], mutants, score, output_path: str):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame.from_dict({"WT": wt_seqs, "mutants": mutants, "score": score})
    df.to_csv(output_path, index=False)


def main(args):
    # Set up multiprocessing
    # mp.set_start_method("spawn", force=True)
    # os.environ["OMP_NUM_THREADS"] = "1"

    # Init env stuffs
    set_seed(args.seed) if args.set_seed_only else enable_full_deterministic(args.seed)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = 'true'
    cpu_device = torch.device("cpu")
    mutation_device = torch.device("cpu" if args.devices == "-1" else f"cuda:{args.devices}")

    # Get sequences
    sequences, targets = None, None
    if args.data_file is not None:
        sequences, targets = extract_from_csv(args.data_file, args.population)

    # Init models
    mutation_model, mutation_tokenizer = initialize_mutation_model(args, mutation_device)
    fitness_predictor = intialize_fitness_predictor(args, cpu_device)
    # Init masker
    maskers = initialize_maskers(args)

    # Init procedure
    direct_evo = DiscreteDirectedEvolution2(
        n_steps=args.n_steps,
        population=args.population,
        maskers=maskers,
        mutation_model=mutation_model,
        mutation_tokenizer=mutation_tokenizer,
        fitness_predictor=fitness_predictor,
        conditional_task=not args.unconditional,
        remove_duplications=args.rm_dups,
        k=args.k,
        population_ratio_per_mask=args.population_ratio_per_mask,
        use_gaussian=args.use_gaussian,
        scoring_mirror=args.scoring_mirror,
        batch_size_inference=args.batch_size,
        num_propose_mutation_per_variant=args.num_proposes_per_var,
        verbose=args.verbose,
        mutation_device=mutation_device,
    )

    mutants, fitnesses = direct_evo(args.wt, args.wt_fitness, sequences, targets)
    fitnesses = fitnesses.squeeze(1).numpy().tolist()
    filename = args.save_name or "results_" + os.path.basename(args.data_file[0])
    filepath = os.path.join(args.result_dir, filename)
    save_results([args.wt] * len(mutants), mutants, fitnesses, filepath)

    # final_evos = []
    # start_idx = 0
    # for i, seq in enumerate(sequences[:1]):
    #     evo = direct_evo(seq)
    #     final_evos.append(evo)
    #     if args.save_interval > 0 and i != 0 and i % args.save_interval == 0:
    #         filename = args.save_name or "results_" + os.path.basename(args.csv_file[0])
    #         filepath = os.path.join(args.result_dir, f"batch_{str(i)}_{filename}")
    #         save_results(sequences[start_idx:i + 1], final_evos, filepath)
    #         start_idx = i
    #         final_evos = []

    # sequences = sequences[args.start_index:]

    # final_evos = []
    # full_length = len(sequences)
    # pool = mp.Pool(processes=args.num_processes)
    # if args.save_interval > 0:
    #     sequences = [sequences[i * args.save_interval:(i + 1) * args.save_interval]
    #                  for i in range((len(sequences) + args.save_interval - 1) // args.save_interval)]
    # if isinstance(sequences[0], list):
    #     for i, seqs in enumerate(sequences):
    #         evos = pool.map(direct_evo, seqs)
    #         filename = args.save_name or "results_" + os.path.basename(args.csv_file[0])
    #         filepath = os.path.join(
    #             args.result_dir,
    #             f"batch_{args.start_index + i * args.save_interval}-"
    #             f"{args.start_index + (i + 1) * args.save_interval - 1}_{filename}"
    #             if (i + 1) * args.save_interval < full_length
    #             else f"batch_{i * args.save_interval}-{full_length - 1}_{filename}"
    #         )
    #         save_results(seqs, evos, filepath)
    #         final_evos.extend(evos)
    #         evos = []
    # else:
    #     final_evos = pool.map(direct_evo, sequences)

    # pool.close()
    # pool.join()

    # filename = args.save_name or "results_" + os.path.basename(args.csv_file[0])
    # filepath = os.path.join(args.result_dir, filename)
    # save_results(tmp_seqs, final_evos, filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
