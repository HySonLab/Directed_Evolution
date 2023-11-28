import sys
import pandas as pd
import os
from de.predictors.tranception.utils.scoring_utils import get_mutated_sequence


def get_substrate(type):
    if type is None or type == "A":
        return "Acetamide"
    elif type == "I":
        return "Isobutyramide"
    elif type == "P":
        return "Propionamide"
    else:
        raise ValueError(f"Substrate type {type} is not supported. Choices are 'A', 'I', and 'P'")


def get_aa_sequence(filepath):
    with open(filepath, "r") as f:
        seq = f.readlines()[0].strip()
    return seq


def generate_data(wt_seq, data_file):
    df = pd.read_csv(data_file, sep="\t")
    # preprocess df
    df = df[df["mutation"] != "*"]
    df = df[df["normalized_fitness"] != "NS"]

    sequences = []
    fitnesses = []

    for i in range(len(df)):
        # get vars
        loc = df["location"].iloc[i]
        wt_aa = wt_seq[loc - 1]
        new_aa = df["mutation"].iloc[i]
        fitness = float(df["normalized_fitness"].iloc[i])

        mut = wt_aa + str(loc) + new_aa
        mut_seq = get_mutated_sequence(wt_seq, mut)

        sequences.append(mut_seq)
        fitnesses.append(fitness)

    return {"sequence": sequences, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    substrate = get_substrate(sys.argv[2])
    seq_file = os.path.join(data_dir, "amiE_reference_sequence.txt")
    data_file = os.path.join(data_dir, f"amiESelectionFitnessData_{substrate}.txt")
    out_file = os.path.join(data_dir, f"amiE_{substrate}.csv")

    # Get protein sequence
    wt_seq = get_aa_sequence(seq_file)

    # Generate data
    seq2fit = generate_data(wt_seq, data_file)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    # Drop duplications
    mut_df.drop_duplicates(subset="sequence", inplace=True, ignore_index=True)
    mut_df.to_csv(out_file, index=False)
