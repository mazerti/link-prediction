import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """Main function."""
    args = parse_args()
    folder = os.path.dirname(args.results)
    df = pd.read_csv(args.results)
    log = bool(args.log)

    plot_distribution(df, "l2-score", folder, log)
    plot_distribution(df, "dot-product-score", folder, log)
    plot_distribution_positives(df, "l2-rank", folder, log)
    plot_distribution_positives(df, "dot-product-rank", folder, log)


def plot_distribution_positives(df: pd.DataFrame, column: str, folder: str, log: bool):
    positive_ranks = df.loc[df["label"] == 1, column].to_frame()

    plot_distribution(positive_ranks, column, folder, log)


def plot_distribution(df, column, folder, log: bool):
    plt.figure(figsize=(10, 5))
    plt.hist(df[column], bins=100, alpha=0.7, log=log)
    plt.title(f"{column} Distribution\n({folder})")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.savefig(
        os.path.join(folder, f"{column}_distribution{"-log" if log else ""}.png")
    )


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Link Prediction for Temporal Interaction Network"
    )
    parser.add_argument(
        "results",
        type=str,
        help="Path to the file containing the results.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Add this tag if you want to use a log scale for the y-axis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
