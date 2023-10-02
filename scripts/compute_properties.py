import argparse
import numpy as np
import pandas as pd
import os
from modlamp.descriptors import GlobalDescriptor
from typing import List
from de.common.io_utils import read_fasta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_files",
                        type=str,
                        nargs="+",
                        help="List of data files.")
    parser.add_argument("--output_name",
                        type=str,
                        default="data",
                        help="Name of output file.")
    args = parser.parse_args()
    return args


def get_sequence_from_fasta(fasta_file: str):
    fasta_seqs = read_fasta(fasta_file)
    return list(set(fasta_seqs.values()))   # remove duplications


def compute_properties(seqs: List[str]) -> np.ndarray:
    desc = GlobalDescriptor(seqs)
    desc.instability_index()
    insta_idx = desc.descriptor.ravel()

    desc.boman_index()
    boman_idx = desc.descriptor.ravel()

    desc.charge_density()
    charge = desc.descriptor.ravel()

    return insta_idx, boman_idx, charge


def write_csv(filepath, header: List[str], *properties):
    data = {h: p for h, p in zip(header, properties)}
    pd.DataFrame(data).to_csv(filepath, index=False)


def main(args):
    header = ["sequence", "instability_index", "Boman_index", "charge_density"]
    sequences = []
    for filepath in args.data_files:
        seqs = get_sequence_from_fasta(filepath)
        sequences.extend(seqs)

    insta_idx, boman_idx, charge_density = compute_properties(sequences)
    save_path = os.path.join(
        os.path.dirname(filepath),
        f"{args.output_name}.csv"
    )
    write_csv(save_path, header, sequences, insta_idx, boman_idx, charge_density)


if __name__ == "__main__":
    args = parse_args()
    main(args)
