"""Plot latency boxplot from benchmark CSV results.

Usage:
    python plot_latency.py --csv results.csv --outfig figures/latency_box.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font="Poppins")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outfig", default="figures/latency_box.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.boxplot(x="store", y="lat_ms", data=df, ax=ax, palette="Set2")
    ax.set_xlabel("")
    ax.set_ylabel("P95 Latency (ms)")
    fig.tight_layout()
    Path(args.outfig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outfig)
    print(f"Wrote {args.outfig}")


if __name__ == "__main__":
    main()
