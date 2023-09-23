import argparse
import numpy as np
import pandas as pd
import os
from pickle import dump
from de.predictors.gaussian_process import GaussianProcess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="K-mer.")
    parser.add_argument("--csv_file", type=str, help="Path to csv data.")
    parser.add_argument("--save_dir", type=str, default="../exps", help="Checkpoint save dir.")
    args = parser.parse_args()
    return args


def extract_from_csv(filepath):
    df = pd.read_csv(filepath)
    insta_idx = df.instability_index.tolist()
    boman_idx = df.Boman_index.tolist()
    seqs = df.sequence.tolist()
    targets = np.array([[y1, y2] for y1, y2 in zip(insta_idx, boman_idx)],
                       dtype=np.float32)
    return seqs, targets


def main(args):
    gp = GaussianProcess(k=args.k)
    seqs, targets = extract_from_csv(args.csv_file)
    print("Start training Gaussian Process model...")
    gp.fit(seqs, targets)
    print("Finish!!")
    save_path = os.path.join(
        args.save_dir,
        os.path.basename(args.csv_file).replace(".csv", ".pkl")
    )
    print(f"Saving model to {save_path}")
    with open(save_path, "wb") as f:
        dump(gp, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
