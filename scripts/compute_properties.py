import argparse
import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor
from typing import List
from de.common.io_utils import read_fasta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_files",
                        type=str,
                        nargs="+",
                        help="List of data files.")
    args = parser.parse_args()
    return args


def get_sequence_from_fasta(fasta_file: str):
    fasta_seqs = read_fasta(fasta_file)
    return list(fasta_seqs.values())


def compute_properties(fasta_file) -> np.ndarray:
    seqs = get_sequence_from_fasta(fasta_file)
    desc = GlobalDescriptor(seqs)
    desc.instability_index()
    desc.boman_index(append=True)
    return seqs, desc.descriptor[:, 0], desc.descriptor[:, 1]


def write_csv(filepath, header: List[str], *properties):
    data = {h: p for h, p in zip(header, properties)}
    pd.DataFrame(data).to_csv(filepath, index=False)


def main(args):
    header = ["sequence", "instability_index", "Boman_index"]
    for filepath in args.data_files:
        seqs, insta_idx, boman_idx = compute_properties(filepath)
        save_path = filepath.replace(".fasta", ".csv")
        write_csv(save_path, header, seqs, insta_idx, boman_idx)


if __name__ == "__main__":
    args = parse_args()
    main(args)
