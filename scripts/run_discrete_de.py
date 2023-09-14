import argparse
import os
import pandas as pd
import torch
import torch.multiprocessing as mp
import traceback
from typing import List, Union, Tuple
from transformers import PreTrainedTokenizerFast
from de.common.io_utils import read_fasta
from de.common.utils import set_seed, enable_full_deterministic
from de.directed_evolution import DiscreteDirectedEvolution
from de.samplers.maskers import RandomMasker, ImportanceMasker
from de.samplers.models.esm import ESM2
from de.predictors.tranception.model import TranceptionLMHeadModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file",
                        type=str,
                        default="../data/uniprot_sprot.fasta",
                        help="Path to fasta file.")
    parser.add_argument("--vocab_file",
                        type=str,
                        default="../data/vocabs/vocab_1.txt",
                        help="Path to vocabulary file.")
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
    parser.add_argument("--population_ratio_per_mask",
                        nargs="+",
                        type=float,
                        help="Population ratio to run per masker.")
    parser.add_argument("--mutation_model_ckpt",
                        type=str,
                        help="Path to checkpoint of mutation model.")
    parser.add_argument("--edit_range",
                        nargs="+",
                        type=int,
                        help="Allowed region to mask sequence.")
    parser.add_argument("--pretrained_mutation_path",
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
    parser.add_argument("--mutation_ckpt",
                        type=str,
                        help="Path to the checkpoint of mutation model (Protein LM).")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Whether to display output.")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Device(s) to run directed evoltion, separated by comma.")
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
    args = parser.parse_args()
    return args


def get_sequence_from_fasta(fasta_file: str, max_seq_len: int) -> List[str]:
    seqs = read_fasta(fasta_file, max_seq_length=max_seq_len)
    seqs = seqs.values()
    return list(seqs)


# TODO: re-write this function (fit with ESM2)
def initialize_mutation_model(args, device: torch.device):
    model = ESM2(vocab_file=args.vocab_file,
                 pretrained_model_name_or_path=args.pretrained_mutation_path)
    tokenizer = model.tokenizer
    model = model.to(device)
    return model, tokenizer


def initialize_maskers(args):
    imp_masker = ImportanceMasker(use_full_dataset=False,
                                  max_subs=args.num_masked_tokens,
                                  low_importance_mask=not args.mask_high_importance)
    rand_masker = RandomMasker(max_subs=args.num_masked_tokens)

    return [imp_masker, rand_masker]


def intialize_fitness_predictor(tokenizer_file: str,
                                model_size: str = "Small",
                                device: Union[str, torch.device] = "cpu",):
    model_name = f"PascalNotin/Tranception_{model_size}"
    model = TranceptionLMHeadModel.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model.config.tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]"
    )
    model = model.to(device)
    return model


def save_results(evos: List[Tuple[str, float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    mutants, score = zip(*evos)
    df = pd.DataFrame.from_dict({"mutants": list(mutants), "score": list(score)})
    df.to_csv(os.path.join(output_dir, "result.csv"))


def main(args):
    # Get sequences
    sequences = get_sequence_from_fasta(args.fasta_file, args.max_sequence_length)

    # Init env stuffs
    set_seed(args.seed) if args.set_seed_only else enable_full_deterministic(args.seed)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = 'true'
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # device = "cpu" if args.devices == "-1" else "cuda"
    device = torch.device("cpu")

    # Init models
    mutation_model, mutation_tokenizer = initialize_mutation_model(args, device)
    fitness_predictor = intialize_fitness_predictor(args.tokenizer_file,
                                                    args.tranception_type)

    # Init masker
    maskers = initialize_maskers(args)

    # Init procedure
    direct_evo = DiscreteDirectedEvolution(
        n_steps=args.n_steps,
        population=args.population,
        maskers=maskers,
        mutation_model=mutation_model,
        mutation_tokenizer=mutation_tokenizer,
        fitness_predictor=fitness_predictor,
        batch_size_inference=args.batch_size,
        num_propose_mutation_per_variant=args.num_proposes_per_var,
        k=args.k,
        population_ratio_per_mask=args.population_ratio_per_mask,
        edit_range=args.edit_range,
        mutation_ckpt_path=args.mutation_ckpt,
        verbose=args.verbose,
        device=device
    )

    mp.set_start_method("spawn", force=True)

    try:
        pool = mp.Pool(args.num_processes)
        evos = pool.map(direct_evo, sequences)
    except Exception as e:
        print(traceback.format_exc())
        print("Main Pool Error:", e)
    except KeyboardInterrupt:
        exit()
    finally:
        pool.terminate()
        pool.join()

    save_results(evos, args.result_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
