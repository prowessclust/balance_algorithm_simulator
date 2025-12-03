#!/usr/bin/env python3
"""
compare_metrics_curves.py

Combine the metrics_<agg>_*.csv files produced by balance_with_metrics.py
and plot comparison curves for Max.TER, Consensus Error, and AvgBenign accuracy.

Usage:
  python compare_metrics_curves.py --csvs metrics_balance_clients10_mal3_atksign_flip.csv \
                                   metrics_fedavg_clients10_mal3_atksign_flip.csv \
                                   metrics_trim_clients10_mal3_atksign_flip.csv
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def load_metrics_csv(path):
    """Return rounds, dict(metric -> list of values)."""
    rounds = []
    data = {"max_ter": [], "max_asr": [], "consensus": [], "avg_benign": [], "avg_all": []}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            for k in data.keys():
                data[k].append(float(row[k]))
    return np.array(rounds), data


def extract_label(csv_path):
    fname = os.path.basename(csv_path)
    for p in ["balance", "fedavg", "trim", "median", "krum"]:
        if p in fname:
            return p
    return fname.split(".")[0]


def plot_metric_comparison(metric_key, csv_paths, title, ylabel, out_dir):
    plt.figure(figsize=(8, 5))
    for csv_file in csv_paths:
        rounds, data = load_metrics_csv(csv_file)
        label = extract_label(csv_file)
        plt.plot(rounds, data[metric_key], label=label)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{metric_key}_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved", metric_key, "comparison to", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True, help="List of metrics CSVs to compare")
    parser.add_argument("--out", default="plots", help="Output directory")
    args = parser.parse_args()

    metrics = [
        ("max_ter", "Max Test Error Rate (Max.TER)", "Max TER"),
        ("consensus", "Consensus Error (Benign Models)", "Consensus Error"),
        ("avg_benign", "Average Benign Accuracy", "Accuracy"),
    ]

    for metric_key, title, ylabel in metrics:
        plot_metric_comparison(metric_key, args.csvs, title, ylabel, args.out)


if __name__ == "__main__":
    main()
