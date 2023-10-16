import pandas as pd
import sys
import os
from Bio.Seq import translate


def get_aa_sequence(filepath: str):
    with open(filepath, "r") as f:
        content = f.readlines()
    dna_seq = content[-1]
    prot_seq = translate(dna_seq, to_stop=True)
    return prot_seq


def mutant2seq(wt_seq: str, mutant: str):
    if mutant == "":
        return wt_seq
    elif "*" in mutant:
        return None
    else:
        seq = list(wt_seq)
        muts = mutant.split(":")
        for mut in muts:
            aa_org, pos, aa_new = mut[1], int(mut[2:-1]), mut[-1]
            if aa_org != wt_seq[pos]:
                raise ValueError(f"{aa_org} is different from wt_seq[{pos}].")
            seq[pos] = aa_new

        return "".join(seq)


def generate_data(wt_seq: str, df: pd.DataFrame):
    df["aaMutations"].fillna("", inplace=True)
    mutants = df["aaMutations"].tolist()
    fitness = df["medianBrightness"].tolist()
    variants = []
    fitnesses = []
    for mut, fit in zip(mutants, fitness):
        variant = mutant2seq(wt_seq, mut)
        if variant is not None:
            variants.append(variant)
            fitnesses.append(fit)

    return {"sequence": variants, "fitness": fitnesses}


if __name__ == "__main__":
    # Files
    data_dir = sys.argv[1]
    seq_file = os.path.join(data_dir, "avGFP_reference_sequence.fa")
    data_file = os.path.join(data_dir, "amino_acid_genotypes_to_brightness.tsv")
    out_file = os.path.join(data_dir, "avGFP.csv")

    # Convert DNA to protein sequence
    wt_seq = get_aa_sequence(seq_file)

    # Generate data
    df = pd.read_csv(data_file, sep="\t")
    seq2fit = generate_data(wt_seq, df)
    mut_df = pd.DataFrame.from_dict(seq2fit)
    mut_df.to_csv(out_file, index=False)
