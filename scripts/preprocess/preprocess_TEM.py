import sys
import numpy as np
import pandas as pd
import os
from de.common.utils import get_mutated_sequence


def get_aa_sequence(filepath):
    with open(filepath, "r") as f:
        seq = f.readlines()[0].strip()
    return seq


def generate_data(wt_seq, data_file):
    df = pd.read_csv(data_file, sep="\t")
    # df.dropna(inplace=True, ignore_index=True)
    df["real_loc"] = df["location"].apply(lambda x: x - 3)

    sequences = []
    fitnesses = []

    for i in range(len(df)):
        # get vars
        loc = df["real_loc"].iloc[i]
        wt_aa = df["wt_aa"].iloc[i]
        new_aa = df["new_aa"].iloc[i]
        fitness = df["fitness"].iloc[i]

        if np.isnan(fitness):
            continue

        if wt_seq[loc] != wt_aa:
            print(f"i = {i}")
            print(f"loc = {loc}")
            print(df.iloc[i])
            print(f"wt_seq[{loc}] = {wt_seq[loc]}")
            print(f"wt_aa = {wt_aa}")
            raise ValueError(f"Position {loc + 1} of WT sequence is {wt_seq[loc]}, not {wt_aa}")
        mut = wt_aa + str(loc + 1) + new_aa
        mut_seq = get_mutated_sequence(wt_seq, mut)

        sequences.append(mut_seq)
        fitnesses.append(fitness)

    return {"sequence": sequences, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    seq_file = os.path.join(data_dir, "TEM_reference_sequence.txt")
    data_file = os.path.join(data_dir, "TEM_mutation.tsv")
    out_file = os.path.join(data_dir, "TEM.csv")

    # Get protein sequence
    wt_seq = get_aa_sequence(seq_file)

    # Generate data
    seq2fit = generate_data(wt_seq, data_file)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    # Drop duplications
    mut_df.drop_duplicates(subset="sequence", inplace=True, ignore_index=True)
    mut_df.to_csv(out_file, index=False)
