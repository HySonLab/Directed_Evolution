import argparse
import os
import glob
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name",
                        type=str,
                        help="Name of dataset to merge.")
    parser.add_argument("--root_dir",
                        type=str,
                        default="/home/thanhtvt1/workspace/Directed_Evolution/exps/results",
                        help="Path to root folder.")
    parser.add_argument("--delete_files",
                        action="store_true",
                        help="Whether to delete files after merging.")
    args = parser.parse_args()
    return args


def main(args):
    file_pattern = f"batch_*_{args.name}*.csv"
    # get list of csv files matching the pattern
    files = glob.glob(os.path.join(args.root_dir, file_pattern))
    # sort file
    files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split('-')[0]))

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged data
    merged_df.to_csv(os.path.join(args.root_dir, f"result_{args.name}.csv"), index=False)

    # delete file if needed
    if args.delete_files:
        for file in files:
            os.remove(file)

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
