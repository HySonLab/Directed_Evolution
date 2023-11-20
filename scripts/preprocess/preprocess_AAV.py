import sys
import numpy as np
import pandas as pd
import os


def get_aa_sequence(filepath):
    with open(filepath, "r") as f:
        seq = f.readlines()[0].strip()
    return seq


def generate_data(data_file):
    df = pd.read_csv(data_file)

    # preprocess data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    sequences = [seq.upper() for seq in df["sequence"].to_list()]
    fitnesses = df["viral_selection"].to_list()

    return {"sequence": sequences, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    seq_file = os.path.join(data_dir, "AAV_reference_sequence.txt")
    data_file = os.path.join(data_dir, "allseqs_20191230.csv")
    out_file = os.path.join(data_dir, "AAV.csv")

    # Generate data
    seq2fit = generate_data(data_file)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    # Drop duplications
    mut_df.drop_duplicates(subset="sequence", inplace=True, ignore_index=True)
    mut_df.to_csv(out_file, index=False)
