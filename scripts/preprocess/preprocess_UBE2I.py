import sys
import numpy as np
import pandas as pd
import os
from de.common.utils import get_mutated_sequence


def get_aa_sequence(filepath):
    with open(filepath, "r") as f:
        seq = f.readlines()[0].strip()
    return seq


def generate_data(wt_seq, data_file1, data_file2):
    df1 = pd.read_csv(data_file1)
    df2 = pd.read_csv(data_file2)

    sequences = []
    fitnesses = []

    for i in range(len(df1)):
        mut = df1["mut"].iloc[i]
        mut_seq = get_mutated_sequence(wt_seq, mut)
        if np.isnan(df1["screen.score"].iloc[i]) and np.isnan(df2["screen.score"].iloc[i]):
            fitness = (df1["joint.score"].iloc[i] + df2["joint.score"].iloc[2]) / 2
        else:
            fitness = df1["screen.score"].iloc[i] or df2["screen.score"].iloc[i]

        sequences.append(mut_seq)
        fitnesses.append(fitness)

    return {"sequence": sequences, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    seq_file = os.path.join(data_dir, "UBE2I_reference_sequence.txt")
    data_file1 = os.path.join(data_dir, "UBE2I_scores.csv")
    data_file2 = os.path.join(data_dir, "UBE2I_flipped_scores.csv")
    out_file = os.path.join(data_dir, "UBE2I.csv")

    # Get protein sequence
    wt_seq = get_aa_sequence(seq_file)

    # Generate data
    seq2fit = generate_data(wt_seq, data_file1, data_file2)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    # Drop duplications
    mut_df.drop_duplicates(subset="sequence", inplace=True, ignore_index=True)
    mut_df.to_csv(out_file, index=False)
