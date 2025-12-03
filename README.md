# Decentralized BALANCE — Simulation, Baselines, Metrics & Plots

A compact, reproducible simulation of decentralized peer-to-peer learning implementing the BALANCE aggregation rule, multiple baseline aggregators (FedAvg, Trimmed-mean, Median, Krum), and common attack scenarios (sign_flip, random_noise, label_flip). The script trains lightweight CNN clients on MNIST, runs neighbor-based aggregation on a graph topology, and saves metrics and plots locally.

---

## Repository contents (key files)
- `balance_with_metrics.py` — main simulation script
- `README.md` — this file
- `.gitignore` — ignores data, results, models, secrets, and common large files
- `results/` — timestamped folders containing run CSVs and plots (generated locally)

---

## Quick start (project root)
1. Install dependencies:
   pip install torch torchvision numpy matplotlib networkx

2. Run a short demo:
   python3 balance_with_metrics.py --num_clients 6 --rounds 10 --malicious 1 --attack sign_flip --agg balance --seed 2025

3. See outputs in a timestamped folder under `results/`.

---

## Command-line flags (most used)
- `--num_clients` (int, default=10) — number of clients
- `--rounds` (int, default=30) — communication rounds
- `--malicious` (int, default=2) — number of malicious clients
- `--attack` (str, default=sign_flip) — choices: `none`, `random_noise`, `sign_flip`, `label_flip`
- `--agg` (str, default=balance) — choices: `balance`, `fedavg`, `trim`, `median`, `krum`
- `--seed` (int, default=42) — random seed
- `--trim_frac` (float, default=0.2) — for trimmed-mean
- `--krum_f` (int, default=1) — estimated Byzantine count for Krum

---

## High-level script flow
1. Download MNIST (if not cached) and split train/test across clients.
2. Build a Watts–Strogatz graph for client neighbors.
3. Initialize clients with a lightweight CNN (two conv layers; `fc1` is created on first forward).
4. For each round:
   - Each client performs one epoch of local training.
   - Malicious clients craft updates according to the chosen `--attack`.
   - Each client collects neighbor messages and applies the selected aggregator.
   - Clients update local models; metrics are computed and saved.
5. Results saved under `results/<timestamp>_<RUNNAME>/` (CSV + PNG).

---

## Aggregators (summary)
- `balance` — BALANCE: distance-based acceptance + mixing with own model (`alpha`)
- `fedavg` — average neighbor weights
- `trim` — coordinate-wise trimmed mean (`trim_frac`)
- `median` — coordinate-wise median
- `krum` — simplified Krum (selects most central neighbor; `krum_f`)

---

## Attack types
- `none` — honest clients
- `random_noise` — add Gaussian noise to tensors
- `sign_flip` — flip sign and scale (destructive)
- `label_flip` — train on labels shifted by +1 (poisoning)

---

## Outputs (per run)
results/<timestamp>_<RUNNAME>/ contains:
- `metrics_<agg>_clients{N}_mal{M}_atk{attack}.csv` — columns: `round,max_ter,max_asr,consensus,avg_benign,avg_all`
- `asr_per_client_<...>.csv` — `round` + one column per benign client
- `plots/` — PNGs: `asr_per_client.png`, `asr_mean_std.png`, `max_ter_<agg>.png`, `max_asr_<agg>.png`, `consensus_<agg>.png`, `avg_benign_<agg>.png`, `avg_all_<agg>.png`

---

## Example (short demo)
python3 balance_with_metrics.py --num_clients 6 --rounds 10 --malicious 1 --attack sign_flip --agg balance --seed 2025

Console output example (illustrative):
Round 1/10 - Max.TER=0.8450 Max.ASR=0.0125 Consensus=26.5234 AvgBenign=0.1550

---

## Reproducibility & tips
- Use `--seed` for deterministic splits and graph generation.
- Reduce `--num_clients` and `--rounds` for debugging.
- CUDA detection is automatic; note potential nondeterminism across devices.
- If you encounter OOM, reduce client count, model size or run on CPU.

---

## Privacy & repository policy (important)
- Only push code, documentation and small safe examples to GitHub.
- Do NOT commit datasets, results, models, logs, or credentials.
- This repo's `.gitignore` is configured to ignore: `data/`, `results/`, `models/`, `checkpoints/`, `*.csv`, `*.pth`, `*.pt`, `*.pkl`, `.env`, credential files, logs, notebooks, and common binary formats.
- If sensitive files were accidentally committed, remove them with `git rm --cached <path>` and purge history with `git filter-repo` or BFG before pushing.

Safe push checklist (run in project root `/home/priyanka/flwr_app/my-awesome-app`):
1. Configure identity (if not set here):
   git config user.email "you@example.com"
   git config user.name "Your Name"

2. Initialize and preview:
   git init
   git add --all --dry-run

3. Commit .gitignore and essential files first:
   git add .gitignore README.md balance_with_metrics.py
   git commit -m "chore: add .gitignore and initial files"

4. Verify ignored files:
   git status --ignored
   git check-ignore -v path/to/suspect.file

5. Finalize and push:
   git add -A
   git commit -m "chore: initial commit"
   git branch -M main
   git remote add origin https://github.com/<USERNAME>/<REPO>.git
   git push -u origin main

Or use GitHub CLI:
   gh auth login
   gh repo create <REPO> --public --source=. --remote=origin --push

---

## Dependencies
- Python 3.8+
- PyTorch (recommended 1.9+), torchvision
- numpy, matplotlib, networkx

Install:
pip install torch torchvision numpy matplotlib networkx

(Use the appropriate PyTorch wheel for CUDA if needed: https://pytorch.org/)

---

## Extending the code
- Swap dataset (CIFAR-10), replace the model, or add aggregators. Ensure consistent `state_dict` shapes across clients.
- Modify backdoor trigger placement or intensity in `apply_trigger`.
- Save final client models by enabling the save block in the script.

---

## Known limitations & notes
- `fc1` is created on first forward; avoid manipulating state_dicts before that step.
- Krum here is a simplified variant — adapt for production use.
- Flattened state-vector operations may be memory-intensive for many clients.

---
