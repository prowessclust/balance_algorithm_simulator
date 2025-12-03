#!/usr/bin/env python3
"""
robustness_dashboard_multi.py
------------------------------------------------
Auto-detects *all* latest experiment runs under 'results/',
groups them by timestamp (same date/time prefix),
loads their metrics (asr_per_client + metrics_*.csv),
and plots a combined dashboard comparing all aggregators.

Usage:
    python my_awesome_app/robustness_dashboard.py
"""

import os, csv, subprocess, platform
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# consistent colors/styles
COLOR_MAP = {
    "balance": "#1f77b4", "fedavg": "#ff7f0e", "trim": "#2ca02c",
    "median": "#d62728", "krum": "#9467bd",
}
LINE_STYLE = {
    "balance": "-", "fedavg": "--", "trim": "-.",
    "median": ":", "krum": (0, (3, 1, 1, 1))
}

# ------------- helpers -------------

def extract_label(path):
    fname = os.path.basename(path).lower()
    for p in COLOR_MAP:
        if p in fname:
            return p
    return fname.split(".")[0]

def load_asr_csv(path):
    rounds, mat = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rounds.append(int(row[0]))
            mat.append([float(x) for x in row[1:]])
    return np.array(rounds), np.array(mat)

def load_metrics_csv(path):
    rounds, data = [], {"max_ter": [], "consensus": [], "avg_benign": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            data["max_ter"].append(float(row["max_ter"]))
            data["consensus"].append(float(row["consensus"]))
            data["avg_benign"].append(float(row["avg_benign"]))
    return np.array(rounds), data

def parse_run_info(name):
    parts = name.split("_")
    info = {"Timestamp": "?", "Aggregator": "?", "Clients": "?", "Malicious": "?", "Attack": "?"}
    for p in parts:
        if p.lower() in COLOR_MAP:
            info["Aggregator"] = p.lower()
        elif p.startswith("clients"):
            info["Clients"] = p.replace("clients", "")
        elif p.startswith("mal"):
            info["Malicious"] = p.replace("mal", "")
        elif p not in ["results", "plots"] and not p[0].isdigit():
            info["Attack"] = p
    if len(parts) >= 2:
        info["Timestamp"] = parts[0] + "_" + parts[1]
    return info

def open_file(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        elif platform.system == "Linux":
                    # Detect if we are in WSL (Windows Subsystem for Linux)
                    if "microsoft" in platform.uname().release.lower():
                        # Convert /home/... path to Windows-style path (C:\...)
                        wsl_path = subprocess.check_output(["wslpath", "-w", path]).decode().strip()
                        print(f"(WSL detected — opening via Windows: {wsl_path})")
                        subprocess.run(["/mnt/c/Windows/System32/cmd.exe", "/C", "start", "", wsl_path])
                    else:
                        subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"(Could not auto-open PDF: {e})")

def group_runs_by_timestamp(root="results"):
    """Return dict[timestamp] = [run_paths with that timestamp]"""
    if not os.path.exists(root):
        return {}
    runs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    grouped = {}
    for r in runs:
        parts = r.split("_")
        if len(parts) >= 2:
            ts = parts[0] + "_" + parts[1]
            grouped.setdefault(ts, []).append(os.path.join(root, r))
    return grouped

def find_csvs(run_dir):
    asr_csv, metrics_csv = None, None
    for f in os.listdir(run_dir):
        if f.startswith("asr_per_client_") and f.endswith(".csv"):
            asr_csv = os.path.join(run_dir, f)
        elif f.startswith("metrics_") and f.endswith(".csv"):
            metrics_csv = os.path.join(run_dir, f)
    return asr_csv, metrics_csv

# ------------- dashboard -------------

def robustness_dashboard_multi(run_group):
    """
    run_group: list of run directories with same timestamp (different aggregators)
    """
    if not run_group:
        print("⚠️ No runs found.")
        return

    timestamp = os.path.basename(run_group[0]).split("_")[0:2]
    timestamp = "_".join(timestamp)
    out_dir = os.path.join(run_group[0], "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "robustness_dashboard_combined.pdf")
    out_png = os.path.join(out_dir, "robustness_dashboard_combined.png")

    plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(2, 2)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2, "legend.frameon": False})

    # initialize subplots
    ax1 = plt.subplot(gs[0, 0]); ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0]); ax4 = plt.subplot(gs[1, 1])

    for run in run_group:
        info = parse_run_info(os.path.basename(run))
        agg = info["Aggregator"]
        asr_csv, metrics_csv = find_csvs(run)
        if not asr_csv or not metrics_csv:
            continue

        rounds_asr, mat = load_asr_csv(asr_csv)
        mean_asr, std_asr = np.mean(mat, 1), np.std(mat, 1)
        rounds_m, data = load_metrics_csv(metrics_csv)

        color = COLOR_MAP.get(agg, None)
        ls = LINE_STYLE.get(agg, "-")

        ax1.plot(rounds_asr, mean_asr, label=agg.upper(), color=color, linestyle=ls)
        ax1.fill_between(rounds_asr, mean_asr - std_asr, mean_asr + std_asr, color=color, alpha=0.15)

        ax2.plot(rounds_m, data["max_ter"], label=agg.upper(), color=color, linestyle=ls)
        ax3.plot(rounds_m, data["consensus"], label=agg.upper(), color=color, linestyle=ls)
        ax4.plot(rounds_m, data["avg_benign"], label=agg.upper(), color=color, linestyle=ls)

    ax1.set_title("Attack Success Rate (Mean ± Std)")
    ax2.set_title("Max Test Error Rate (Max.TER)")
    ax3.set_title("Consensus Error (Benign Clients)")
    ax4.set_title("Average Benign Accuracy")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Round")
        ax.legend(fontsize=8)

    plt.suptitle(f"Federated Learning Robustness Dashboard — {timestamp}", fontsize=16, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ Combined dashboard saved → {out_pdf}")
    print(f"✅ Combined dashboard saved → {out_png}")
    open_file(out_pdf)

# ------------- entrypoint -------------

if __name__ == "__main__":
    grouped = group_runs_by_timestamp("results")
    if not grouped:
        print("⚠️ No runs found under ./results/. Run some simulations first.")
    else:
        latest_ts = max(grouped.keys())
        run_group = grouped[latest_ts]
        print(f"📂 Detected {len(run_group)} runs under timestamp: {latest_ts}")
        for r in run_group:
            print("   └─", r)
        robustness_dashboard_multi(run_group)



