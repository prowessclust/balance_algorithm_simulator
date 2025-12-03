# Decentralized BALANCE + Baselines + Metrics & Plots

This repository contains `balance_with_metrics.py`, a simulation script for decentralized collaborative learning with the BALANCE aggregation rule, several baseline aggregators, and attack scenarios (backdoors and Byzantine-style attacks). The script trains simple CNN clients on MNIST, runs local training and neighbor-based aggregation in a peer-to-peer graph, and collects metrics and plots (accuracy, attack success rates, consensus error, etc.).

## Features

- Decentralized client simulation (graph topology with neighbors).
- BALANCE aggregation (distance-based acceptance + convex combination).
- Baselines: FedAvg (neighbor averaging), Trimmed-mean, Coordinate-wise Median, Krum.
- Attacks: sign-flip, random-noise, label-flip, none.
- Backdoor trigger generation + per-client Attack Success Rate (ASR).
- Metrics and plots saved per run (CSV and PNG files).
- Timestamped result directories under `results/`.

## Quick Usage

From the repository root:

python3 balance_with_metrics.py --num_clients 10 --rounds 30 --malicious 2 --attack sign_flip --agg balance

Command-line flags:
- `--num_clients` (int, default=10): number of clients in the simulation.
- `--rounds` (int, default=30): number of communication rounds.
- `--malicious` (int, default=2): number of malicious clients.
- `--attack` (str, default=sign_flip): attack type; choices: `none`, `random_noise`, `sign_flip`, `label_flip`.
- `--agg` (str, default=balance): aggregator; choices: `balance`, `fedavg`, `trim`, `median`, `krum`.
- `--seed` (int, default=42): random seed for reproducibility.
- `--trim_frac` (float, default=0.2): trimming fraction for trimmed-mean aggregator.
- `--krum_f` (int, default=1): estimated number of Byzantine for Krum.

Example:

python3 balance_with_metrics.py \
  --num_clients 20 --rounds 50 --malicious 3 \
  --attack random_noise --agg trim --trim_frac 0.25 --seed 1234

## What the script does (overview)

1. Downloads MNIST dataset (if necessary).
2. Splits train/test sets across `num_clients` (non-iid/data-partitioning is simple uniform split).
3. Builds a Watts–Strogatz graph to model client neighbors.
4. Initializes clients (each has a lightweight CNN with dynamic fc layer).
5. For each round:
   - Each client performs one epoch of local training on its partition.
   - Malicious clients craft messages according to `attack` flag.
   - Each client receives its neighbors' messages and applies the selected aggregator:
     - BALANCE: distance threshold acceptance + convex combination with own model.
     - FedAvg: plain average of neighbor weights.
     - Trimmed-mean: coordinate-wise trimmed mean.
     - Median: coordinate-wise median.
     - Krum: simplified selection of the most central neighbor model.
   - Clients update their local model with aggregated weights.
   - Metrics are computed (max TER, per-client ASR on backdoor dataset, consensus error, average accuracies).
6. Saves results under a timestamped directory `results/<timestamp>_<RUNNAME>/`:
   - `metrics_<agg>_clients{N}_mal{M}_atk{attack}.csv`
   - `asr_per_client_<agg>_clients{N}_atk{attack}.csv`
   - `plots/` with PNGs:
     - `asr_per_client.png`, `asr_mean_std.png`
     - `max_ter_<agg>.png`, `max_asr_<agg>.png`, `consensus_<agg>.png`, `avg_benign_<agg>.png`, `avg_all_<agg>.png`

## Script-specific details

- Main script: `balance_with_metrics.py`
- Client implementation: `BalanceClient` class
  - Parameters: `alpha` (mixing with own model), `gamma` (distance threshold multiplier), `kappa` (decay for distance threshold).
  - `rounds` is passed to the client to compute an adaptive threshold.
  - Local training uses SGD, lr=0.01, batch size 32, CrossEntropyLoss.
- Model: `CNN` — two conv layers + dynamically-initialized fc layer (`fc1` built on first forward pass).
- Backdoor:
  - `apply_trigger` writes a small square patch to the top-left corner of the input images.
  - `make_backdoor_testset` builds a triggered test batch with a fixed target label.

## Attack types

- `none`: normal behavior.
- `random_noise`: malicious clients add Gaussian noise to each tensor (std ~1.0).
- `sign_flip`: malicious clients flip sign and scale by -10.0 (strong destructive update).
- `label_flip`: malicious clients train on labels shifted by +1 mod 10 (poisoned training labels).

## Aggregation methods

- `balance`: custom BALANCE aggregator implemented per-client using distance-based acceptance of neighbor messages. Accepted messages are averaged then mixed with the client's own weights using `alpha`.
- `fedavg`: average of received neighbor models.
- `trim`: coordinate-wise trimmed mean (uses `trim_frac`).
- `median`: coordinate-wise median.
- `krum`: simplified Krum that picks the most central neighbor based on pairwise distances with parameter `krum_f` (estimated number of Byzantines).

## Output CSVs & Plots

- Metrics CSV columns: `round, max_ter, max_asr, consensus, avg_benign, avg_all`.
  - `max_ter` — maximum test error rate across benign clients at that round.
  - `max_asr` — maximum attack success rate across benign clients at that round.
  - `consensus` — mean squared distance of benign clients to mean benign model (consensus error).
  - `avg_benign` — mean accuracy across benign clients.
  - `avg_all` — mean accuracy across all clients (including malicious).
- ASR CSV has first column `round` and one column per benign client: the ASR values per round.
- Plots are saved as PNGs and include per-client ASR, mean±std ASR, max TER, max ASR, consensus error, and average accuracies.

## Reproducibility & tips

- Set `--seed` for deterministic dataset splits and graph generation.
- Use smaller `--num_clients` and `--rounds` for debugging locally (or comment out the MNIST downloads if already cached).
- GPU: script will use CUDA if available. For reproducibility across CPU/GPU runs, note that some PyTorch operations can be non-deterministic; setting additional PyTorch deterministic flags may help.
- OOM: the flattened state vector operations (for Krum and aggregation functions) may use memory. Reduce `num_clients`, lower model size, or run on CPU for large runs.
- If training is too slow, reduce the number of local batches (smaller batch size), or reduce model complexity.

## Extending / Modifying

- Dataset: replace MNIST dataset loading with CIFAR-10 or a custom dataset (ensure input channels and model adjusted).
- Model: replace `CNN` with a different architecture; note `state_dict_to_vector` / `vector_to_state_dict` expect consistent shapes across clients.
- Aggregators: add or tune aggregator functions; ensure they accept a dict of received state_dicts.
- Backdoor: modify `apply_trigger` to place triggers in different positions or with different intensities.
- Save models: uncomment the code block near the end of the script to save each client's final model.

## Dependencies

Minimum tested setup:
- Python 3.8+
- PyTorch (1.9+)
- torchvision
- numpy
- matplotlib
- networkx

Install with pip:

pip install torch torchvision numpy matplotlib networkx

(Use the appropriate CUDA-enabled PyTorch wheel for GPU support; see https://pytorch.org/.)

## Known limitations & notes

- The model's `fc1` is dynamically created on first forward pass; this works in the script but be careful when manipulating state_dicts before `fc1` is created.
- `state_dict_to_vector` iterates the state dict keys in insertion order — Python 3.7+ dict ordering ensures consistent behavior across runs within the same code base, but ensure all clients have the same key ordering.
- Krum implementation here is simplified — it chooses one neighbor model rather than performing a weighted multi-model selection used in some Krum variants.
- The BALANCE threshold uses magnitude-decay with `gamma` and `kappa`; tune them to the desired strictness in message acceptance.

## Example experimental suggestions

- Compare `balance` vs `median` vs `trim` under `sign_flip` attacks with varying `malicious` counts.
- Sweep `gamma` and `kappa` to observe how fast the threshold shrinks and the resulting resilience to attacks.
- Try `label_flip` attack and measure global accuracy degradation vs backdoor ASR.

## License & Contact

- License: add your preferred license (MIT recommended if open-source).
- Contact: maintainers or author (add email or GitHub handle).

---

If you'd like, I can:
- add this README.md file directly into the repository,
- or expand the README with example plots and sample outputs from a sample run,
- or produce a short Jupyter notebook that runs a single short simulation and displays outputs inline.

Tell me which you'd prefer and I will proceed.


Readme.md 2
```markdown
# Decentralized BALANCE + Baselines + Metrics & Plots

This repository contains `balance_with_metrics.py`, a simulation script for decentralized collaborative learning with the BALANCE aggregation rule, several baseline aggregators, and attack scenarios (backdoors and Byzantine-style attacks). The script trains simple CNN clients on MNIST, runs local training and neighbor-based aggregation in a peer-to-peer graph, and collects metrics and plots (accuracy, attack success rates, consensus error, etc.).

## Features

- Decentralized client simulation (graph topology with neighbors).
- BALANCE aggregation (distance-based acceptance + convex combination).
- Baselines: FedAvg (neighbor averaging), Trimmed-mean, Coordinate-wise Median, Krum.
- Attacks: sign-flip, random-noise, label-flip, none.
- Backdoor trigger generation + per-client Attack Success Rate (ASR).
- Metrics and plots saved per run (CSV and PNG files).
- Timestamped result directories under `results/`.

## Quick Usage

From the repository root:

python3 balance_with_metrics.py --num_clients 10 --rounds 30 --malicious 2 --attack sign_flip --agg balance

Command-line flags:
- `--num_clients` (int, default=10): number of clients in the simulation.
- `--rounds` (int, default=30): number of communication rounds.
- `--malicious` (int, default=2): number of malicious clients.
- `--attack` (str, default=sign_flip): attack type; choices: `none`, `random_noise`, `sign_flip`, `label_flip`.
- `--agg` (str, default=balance): aggregator; choices: `balance`, `fedavg`, `trim`, `median`, `krum`.
- `--seed` (int, default=42): random seed for reproducibility.
- `--trim_frac` (float, default=0.2): trimming fraction for trimmed-mean aggregator.
- `--krum_f` (int, default=1): estimated number of Byzantine for Krum.

Example:

python3 balance_with_metrics.py \
  --num_clients 20 --rounds 50 --malicious 3 \
  --attack random_noise --agg trim --trim_frac 0.25 --seed 1234

## What the script does (overview)

1. Downloads MNIST dataset (if necessary).
2. Splits train/test sets across `num_clients` (simple uniform split).
3. Builds a Watts–Strogatz graph to model client neighbors.
4. Initializes clients (each has a lightweight CNN with dynamic fc layer).
5. For each round:
   - Each client performs one epoch of local training on its partition.
   - Malicious clients craft messages according to `attack` flag.
   - Each client receives its neighbors' messages and applies the selected aggregator:
     - BALANCE: distance threshold acceptance + convex combination with own model.
     - FedAvg: plain average of neighbor weights.
     - Trimmed-mean: coordinate-wise trimmed mean.
     - Median: coordinate-wise median.
     - Krum: simplified selection of the most central neighbor model.
   - Clients update their local model with aggregated weights.
   - Metrics are computed (max TER, per-client ASR on backdoor dataset, consensus error, average accuracies).
6. Saves results under a timestamped directory `results/<timestamp>_<RUNNAME>/`:
   - `metrics_<agg>_clients{N}_mal{M}_atk{attack}.csv`
   - `asr_per_client_<agg>_clients{N}_atk{attack}.csv`
   - `plots/` with PNGs:
     - `asr_per_client.png`, `asr_mean_std.png`
     - `max_ter_<agg>.png`, `max_asr_<agg>.png`, `consensus_<agg>.png`, `avg_benign_<agg>.png`, `avg_all_<agg>.png`

- or expand the README with example plots and sample outputs from a sample run,

## Script-specific details

- Main script: `balance_with_metrics.py`
- Client implementation: `BalanceClient` class
  - Parameters: `alpha` (mixing with own model), `gamma` (distance threshold multiplier), `kappa` (decay for distance threshold).
  - `rounds` is passed to the client to compute an adaptive threshold.
  - Local training uses SGD, lr=0.01, batch size 32, CrossEntropyLoss.
- Model: `CNN` — two conv layers + dynamically-initialized fc layer (`fc1` built on first forward pass).
- Backdoor:
  - `apply_trigger` writes a small square patch to the top-left corner of the input images.
  - `make_backdoor_testset` builds a triggered test batch with a fixed target label.

## Attack types

- `none`: normal behavior.
- `random_noise`: malicious clients add Gaussian noise to each tensor (std ~1.0).
- `sign_flip`: malicious clients flip sign and scale by -10.0 (strong destructive update).
- `label_flip`: malicious clients train on labels shifted by +1 mod 10 (poisoned training labels).

## Aggregation methods

- `balance`: custom BALANCE aggregator implemented per-client using distance-based acceptance of neighbor messages. Accepted messages are averaged then mixed with the client's own weights using `alpha`.
- `fedavg`: average of received neighbor models.
- `trim`: coordinate-wise trimmed mean (uses `trim_frac`).
- `median`: coordinate-wise median.
- `krum`: simplified Krum that picks the most central neighbor based on pairwise distances with parameter `krum_f` (estimated number of Byzantines).

## Outputs (files & structure)

After a run the script creates a timestamped folder, for example:

results/2025-11-07_18-10_BALANCE_clients10_mal2_sign_flip/
├─ asr_per_client_balance_clients10_atksign_flip.csv
├─ metrics_balance_clients10_mal2_atksign_flip.csv
└─ plots/
   ├─ asr_per_client.png
   ├─ asr_mean_std.png
   ├─ max_ter_balance.png
   ├─ max_asr_balance.png
   ├─ consensus_balance.png
   ├─ avg_benign_balance.png
   └─ avg_all_balance.png

CSV column descriptions:
- metrics CSV: `round, max_ter, max_asr, consensus, avg_benign, avg_all`
- ASR CSV: `round, client_0, client_1, ...` (one column per benign client)

## Example run (short demo)

Run a short quick simulation (for local testing):

python3 balance_with_metrics.py --num_clients 6 --rounds 10 --malicious 1 --attack sign_flip --agg balance --seed 2025

Example console output excerpt (sample):

Round 1/10 - Max.TER=0.8450 Max.ASR=0.0125 Consensus=26.5234 AvgBenign=0.1550
Round 2/10 - Max.TER=0.7320 Max.ASR=0.0300 Consensus=10.2345 AvgBenign=0.2680
Round 5/10 - Max.TER=0.4210 Max.ASR=0.4200 Consensus=2.1234 AvgBenign=0.5790
Round 10/10 - Max.TER=0.2100 Max.ASR=0.8750 Consensus=0.3456 AvgBenign=0.7900

Note: exact numbers vary across seeds, aggregator and attack settings.

## Example metrics CSV (first rows)

Below is a small sample extract of the `metrics_*` CSV a run might produce (values are illustrative):

```csv
round,max_ter,max_asr,consensus,avg_benign,avg_all
1,0.8450,0.0125,26.5234,0.1550,0.1200
2,0.7320,0.0300,10.2345,0.2680,0.2300
3,0.6120,0.0800,5.6789,0.3880,0.3500
4,0.5010,0.1800,3.4567,0.4990,0.4800
5,0.4210,0.4200,2.1234,0.5790,0.5600
6,0.3650,0.5600,1.2345,0.6350,0.6000
7,0.3100,0.7000,0.8901,0.6900,0.6500
8,0.2650,0.8000,0.5678,0.7350,0.7000
9,0.2350,0.8500,0.4567,0.7650,0.7300
10,0.2100,0.8750,0.3456,0.7900,0.7600
```

## Example ASR CSV (first rows)

Excerpt of `asr_per_client_...csv` (one column per benign client) for the same run:

```csv
round,client_0,client_1,client_2,client_3,client_4
1,0.0025,0.0100,0.0150,0.0050,0.0125
2,0.0050,0.0200,0.0300,0.0100,0.0200
3,0.0200,0.0600,0.0700,0.0300,0.0500
...
10,0.8000,0.8700,0.8900,0.8600,0.8750
```

## Example plots (what they show)

- plots/asr_per_client.png — ASR (Attack Success Rate) curve for each benign client across rounds. Useful to see per-client susceptibility to backdoor.
- plots/asr_mean_std.png — Mean ± Std ASR across benign clients.
- plots/max_ter_<agg>.png — Maximum Test Error Rate among benign clients across rounds (higher is worse).
- plots/max_asr_<agg>.png — Maximum ASR across benign clients (higher means more successful backdoor).
- plots/consensus_<agg>.png — Consensus error among benign clients (lower indicates models converge to similar weights).
- plots/avg_benign_<agg>.png — Average benign client accuracy across rounds.
- plots/avg_all_<agg>.png — Average accuracy across all clients including malicious.

Because I cannot include binary images here, below are example small markdown thumbnails referencing the expected filenames (replace with the real images from the results folder after running):

![ASR per client](results/2025-11-07_18-10_BALANCE_clients10_mal2_sign_flip/plots/asr_per_client.png)
![Mean ASR ± Std](results/2025-11-07_18-10_BALANCE_clients10_mal2_sign_flip/plots/asr_mean_std.png)
![Max TER](results/2025-11-07_18-10_BALANCE_clients10_mal2_sign_flip/plots/max_ter_balance.png)

(If you open the run folder in a file browser, you should see these PNG files.)

## Reproducibility & tips

- Set `--seed` for deterministic dataset splits and graph generation.
- Use smaller `--num_clients` and `--rounds` for debugging locally (or rely on cached MNIST to avoid long downloads).
- GPU: script will use CUDA if available. For reproducibility across CPU/GPU runs, note some PyTorch ops may be non-deterministic; set additional PyTorch deterministic flags if required.
- OOM: the flattened state vector operations may allocate large tensors. Reduce `num_clients`, lower model size, or run on CPU for large runs.
- For faster experiments reduce the number of local training epochs or batch size, or use fewer MNIST samples per client (modify splitting).

## Extending / Modifying

- Dataset: replace MNIST dataset loading with CIFAR-10 or a custom dataset (ensure input channels and model adjusted).
- Model: replace `CNN` with a different architecture; note `state_dict_to_vector` and `vector_to_state_dict` expect consistent shapes across clients.
- Aggregators: add or tune aggregator functions; ensure they accept a dict of received state_dicts.
- Backdoor: modify `apply_trigger` to place triggers in different positions or with different intensities.
- Save models: uncomment the code block near the end of the script to save each client's final model.

## Dependencies

Minimum tested setup:
- Python 3.8+
- PyTorch (1.9+)
- torchvision
- numpy
- matplotlib
- networkx

Install with pip:

pip install torch torchvision numpy matplotlib networkx

(Use the appropriate CUDA-enabled PyTorch wheel for GPU support; see https://pytorch.org/.)

## Known limitations & notes

- The model's `fc1` is dynamically created on first forward pass; this works in the script but be careful when manipulating state_dicts before `fc1` is created.
- `state_dict_to_vector` iterates the state dict keys in insertion order — Python 3.7+ dict ordering ensures consistent behavior across runs within the same code base, but ensure all clients have the same key ordering.
- Krum implementation here is simplified — it chooses one neighbor model rather than performing a weighted multi-model selection used in some Krum variants.
- The BALANCE threshold uses magnitude-decay with `gamma` and `kappa`; tune them to the desired strictness in message acceptance.

## Example experiments to run

- Compare `balance` vs `median` vs `trim` under `sign_flip` attacks with varying `malicious` counts.
- Sweep `gamma` and `kappa` to observe how fast the threshold shrinks and the resulting resilience to attacks.
- Try `label_flip` attack and measure global accuracy degradation vs backdoor ASR.

## License & Contact

- License: add your preferred license (MIT recommended if open-source).
- Contact: maintainers or author (add email or GitHub handle).

---

If you'd like, I can:
- add this README.md file directly into the repository,
- or produce a short Jupyter notebook that runs a single short simulation and displays outputs inline.
```