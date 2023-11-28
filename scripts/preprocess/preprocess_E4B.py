import sys
import pandas as pd
import os
from de.predictors.tranception.utils.scoring_utils import get_mutated_sequence


def get_aa_sequence(filepath):
    with open(filepath, "r") as f:
        seq = f.readlines()[0].strip()
    return seq


def generate_data(wt_seq, data_file):
    df = pd.read_csv(data_file, sep="\t")
    df.dropna(subset="log2_ratio", inplace=True)

    sequences = []
    fitnesses = []

    def convert2mutant(mutations):
        context = mutations.split("-")
        locs = [int(loc) for loc in context[0].split(",")]
        aas = context[1].split(",")
        if "*" in aas:
            return None
        mutants = ""
        for loc, aa in zip(locs, aas):
            mutants = mutants + f"{wt_seq[loc]}{loc + 1}{aa}" + ":"
        return mutants[:-1]

    for i in range(len(df)):
        mutations = df["seqID"].iloc[i]
        mutant = convert2mutant(mutations)
        if mutant is None:
            continue
        seq = get_mutated_sequence(wt_seq, mutant)
        sequences.append(seq)
        fitnesses.append(df["log2_ratio"].iloc[i])

    return {"sequence": sequences, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    seq_file = os.path.join(data_dir, "E4B_reference_sequence.txt")
    data_file = os.path.join(data_dir, "1303309110_sd01.tsv")
    out_file = os.path.join(data_dir, "E4B.csv")

    # Get protein sequence
    wt_seq = get_aa_sequence(seq_file)

    # Generate data
    seq2fit = generate_data(wt_seq, data_file)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    # Drop duplications
    mut_df.drop_duplicates(subset="sequence", inplace=True, ignore_index=True)
    mut_df.to_csv(out_file, index=False)
