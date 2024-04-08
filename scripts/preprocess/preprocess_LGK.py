import sys
import pandas as pd
import os
from de.predictors.tranception.utils.scoring_utils import get_mutated_sequence


def get_aa_sequence(filepath):
    with open(filepath, "r") as f:
        seq = f.readlines()[0].strip()
    return seq


def generate_data(wt_seq, data_file):
    df = pd.read_csv(data_file)
    # preprocess df
    df = df[df["Mutation"] != "*"]
    df = df[df["Normalized_ER"] != "NS"]

    sequences = []
    fitnesses = []

    for i in range(len(df)):
        # get vars
        loc = df["Location"].iloc[i]
        wt_aa = wt_seq[loc]
        new_aa = df["Mutation"].iloc[i]
        fitness = float(df["Normalized_ER"].iloc[i])

        mut = wt_aa + str(loc + 1) + new_aa
        mut_seq = get_mutated_sequence(wt_seq, mut)

        sequences.append(mut_seq)
        fitnesses.append(fitness)

    return {"sequence": sequences, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    seq_file = os.path.join(data_dir, "LGK_reference_sequence.txt")
    data_file = os.path.join(data_dir, "raw.csv")
    out_file = os.path.join(data_dir, "LGK.csv")

    # Get protein sequence
    wt_seq = get_aa_sequence(seq_file)

    # Generate data
    seq2fit = generate_data(wt_seq, data_file)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    # Drop duplications
    mut_df.drop_duplicates(subset="sequence", inplace=True, ignore_index=True)
    mut_df.to_csv(out_file, index=False)
