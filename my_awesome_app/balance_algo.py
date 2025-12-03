#!/usr/bin/env python3
"""
balance_with_metrics.py

Decentralized BALANCE + baselines + metrics & plots.

Usage:
  python balance_with_metrics.py --num_clients 10 --rounds 30 --malicious 2 --attack sign_flip --agg balance

Outputs:
  - metrics_<agg>_run.csv
  - plots: max_ter.png, max_asr.png, consensus.png, avg_acc.png
"""

import argparse
import copy
import math
import os
import random
from typing import Dict, List, Tuple
import csv
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# ---------------------------
# Model (dynamic flattening)
# ---------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = None
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 64).to(x.device)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Helpers for state_dict ops
# ---------------------------
def state_dict_to_vector(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = []
    for k in sd:
        tensors.append(sd[k].flatten())
    return torch.cat(tensors)

def vector_to_state_dict(vec: torch.Tensor, template_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    idx = 0
    for k in template_sd:
        numel = template_sd[k].numel()
        out[k] = vec[idx: idx + numel].view_as(template_sd[k]).clone()
        idx += numel
    return out

def sd_distance(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    # Euclidean distance between flattened vectors (on CPU)
    va = state_dict_to_vector(a).cpu()
    vb = state_dict_to_vector(b).cpu()
    return float(torch.norm(va - vb).item())

def sd_magnitude(a: Dict[str, torch.Tensor]) -> float:
    v = state_dict_to_vector(a).cpu()
    return float(torch.norm(v).item())

def sd_mean(list_sd: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # element-wise average
    if len(list_sd) == 0:
        raise ValueError("Empty list for sd_mean")
    out = {}
    for k in list_sd[0].keys():
        stacked = torch.stack([sd[k].cpu() for sd in list_sd], dim=0)
        out[k] = torch.mean(stacked, dim=0)
    return out

# ---------------------------
# Aggregators (per-client)
# ---------------------------
def fedavg_agg(received: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not received:
        return {}
    return sd_mean(list(received.values()))

def trimmed_mean_agg(received: Dict[int, Dict[str, torch.Tensor]], trim_frac: float = 0.2) -> Dict[str, torch.Tensor]:
    # coordinate-wise trimmed mean
    if not received:
        return {}
    keys = list(next(iter(received.values())).keys())
    n = len(received)
    trim_k = int(math.floor(trim_frac * n))
    out = {}
    for k in keys:
        # gather matrix (n x param_elements) may be large; do flatten-per-parameter tensor approach
        stacked = torch.stack([sd[k].flatten().cpu() for sd in received.values()], dim=0)  # n x m
        # compute trimmed mean per coordinate
        if trim_k == 0:
            coord_mean = torch.mean(stacked, dim=0)
        else:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k: n - trim_k, :]
            if trimmed.shape[0] == 0:
                coord_mean = torch.mean(sorted_vals, dim=0)
            else:
                coord_mean = torch.mean(trimmed, dim=0)
        out[k] = coord_mean.view_as(next(iter(received.values()))[k])
    return out

def median_agg(received: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not received:
        return {}
    out = {}
    keys = list(next(iter(received.values())).keys())
    for k in keys:
        stacked = torch.stack([sd[k].flatten().cpu() for sd in received.values()], dim=0)
        med = torch.median(stacked, dim=0).values
        out[k] = med.view_as(next(iter(received.values()))[k])
    return out

def krum_agg(received: Dict[int, Dict[str, torch.Tensor]], f_est: int = 1) -> Dict[str, torch.Tensor]:
    """
    Simplified Krum: choose one received model (the 'most central') to act as aggregate.
    f_est = estimated number of Byzantine among the received.
    """
    if not received:
        return {}
    ids = list(received.keys())
    vectors = {i: state_dict_to_vector(received[i]).cpu() for i in ids}
    n = len(ids)
    scores = {}
    for i in ids:
        # distances to others
        dists = []
        for j in ids:
            if i == j:
                continue
            dists.append(float(torch.norm(vectors[i] - vectors[j]).item()))
        dists_sorted = sorted(dists)
        nb = max(0, n - f_est - 2)
        # sum of nb smallest distances (if nb <= 0, sum all)
        if nb <= 0:
            scores[i] = sum(dists_sorted)
        else:
            scores[i] = sum(dists_sorted[:nb])
    # select argmin score
    best = min(scores, key=lambda x: scores[x])
    return received[best]

# ---------------------------
# BALANCE client (with local train and BALANCE aggregate)
# ---------------------------
class BalanceClient:
    def __init__(
        self,
        cid: int,
        trainset: data.Dataset,
        testset: data.Dataset,
        neighbors: List[int],
        alpha: float = 0.5,
        gamma: float = 0.3,
        kappa: float = 1.0,
        rounds: int = 100,
        is_malicious: bool = False,
        attack_type: str = "none",
        device: str = None,
    ):
        self.cid = cid
        self.neighbors = neighbors
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.rounds = rounds
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNN().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        self.batch_size = 32
        self.trainset = trainset
        self.testset = testset
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

    def local_train(self, epochs: int = 1) -> Dict[str, torch.Tensor]:
        self.model.train()
        # label flip handled here if attacker type
        if self.is_malicious and self.attack_type == "label_flip":
            flipped = []
            for x, y in self.trainset:
                flipped.append((x, (y + 1) % 10))
            loader = data.DataLoader(flipped, batch_size=self.batch_size, shuffle=True)
        else:
            loader = self.trainloader

        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
        return copy.deepcopy(self.model.state_dict())

    def craft_malicious_message(self, own_state: Dict[str, torch.Tensor]):
        if not self.is_malicious or self.attack_type == "none":
            return own_state
        if self.attack_type == "random_noise":
            return {k: v + torch.randn_like(v) * 1.0 for k, v in own_state.items()}
        if self.attack_type == "sign_flip":
            return {k: v * -10.0 for k, v in own_state.items()}
        if self.attack_type == "label_flip":
            return own_state
        # default
        return {k: v + torch.randn_like(v) * 0.5 for k, v in own_state.items()}


    def balance_aggregate(self, received: Dict[int, Dict[str, torch.Tensor]], t: int) -> Dict[str, torch.Tensor]:
        own_w = copy.deepcopy(self.model.state_dict())
        mag_own = sd_magnitude(own_w)
        lam = (t / max(1, self.rounds))
        thresh = self.gamma * math.exp(-self.kappa * lam) * (mag_own + 1e-12)
        accepted = []
        for j, wj in received.items():
            dist = sd_distance(own_w, wj)
            if dist <= thresh:
                accepted.append(wj)
        if not accepted:
            return own_w
        agg = sd_mean(accepted)
        new_w = {}
        for k in own_w.keys():
            new_w[k] = self.alpha * own_w[k].to(self.device) + (1.0 - self.alpha) * agg[k].to(self.device)
        return new_w

    def test_accuracy(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / max(1, total)

    def predict_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x_batch = x_batch.to(self.device)
            out = self.model(x_batch)
            return torch.argmax(out, dim=1).cpu()

# ---------------------------
# Backdoor trigger utilities
# ---------------------------
def apply_trigger(images: torch.Tensor, trigger_size: int = 3, value: float = 1.0):
    imgs = images.clone()
    imgs[:, :, :trigger_size, :trigger_size] = value  # works fine even for RGB
    return imgs


def make_backdoor_testset(testset, trigger_size=3, target_label=0, n_samples=500):
    # sample n_samples from testset, apply trigger and assign target_label
    xs = []
    ys = []
    rng = np.random.default_rng()
    idxs = rng.choice(len(testset), size=min(n_samples, len(testset)), replace=False)
    for i in idxs:
        x, y = testset[i]
        xs.append(x)
        ys.append(target_label)
    xs = torch.stack(xs, dim=0)
    xs = apply_trigger(xs, trigger_size=trigger_size)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

# ---------------------------
# Metrics & plotting helpers
# ---------------------------
def compute_max_ter(clients: List[BalanceClient], malicious_ids: set) -> float:
    # compute test error rate per benign client and return maximum
    ers = []
    for i, c in enumerate(clients):
        if i in malicious_ids:
            continue
        acc = c.test_accuracy()
        ers.append(1.0 - acc)
    return float(max(ers)) if ers else 0.0

def compute_asr_per_client(clients, malicious_ids, backdoor_data):
    # backdoor_data: (xs, ys_target)
    xs, ys_target = backdoor_data
    asr_per_client = []
    for i, c in enumerate(clients):
        if i in malicious_ids:
            continue
        preds = c.predict_batch(xs)
        asr = float((preds == ys_target).sum().item()) / len(ys_target)
        asr_per_client.append((i, asr))
    return asr_per_client


def save_asr_curves(asr_history, rounds_list, output_dir="plots"):
    """
    asr_history: dict[client_id] = list of asr values per round
    """
    plt.figure(figsize=(8,5))
    for cid, vals in asr_history.items():
        plt.plot(rounds_list, vals, label=f"C{cid}")
    plt.xlabel("Round")
    plt.ylabel("ASR")
    plt.title("Attack Success Rate per Benign Client")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/asr_per_client.png")
    plt.close()

    # Aggregate mean±std plot
    arr = np.array(list(asr_history.values()))
    mean_asr = np.mean(arr, axis=0)
    std_asr = np.std(arr, axis=0)
    plt.figure(figsize=(8,5))
    plt.plot(rounds_list, mean_asr, label="Mean ASR", color="blue")
    plt.fill_between(rounds_list, mean_asr-std_asr, mean_asr+std_asr, color="blue", alpha=0.2)
    plt.xlabel("Round")
    plt.ylabel("ASR")
    plt.title("Mean ± Std ASR (Benign Clients)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/asr_mean_std.png")
    plt.close()


def compute_consensus_error(clients: List[BalanceClient], malicious_ids: set) -> float:
    # mean over benign of ||w_i - mean_w_benign||^2
    benign_sds = [clients[i].model.state_dict() for i in range(len(clients)) if i not in malicious_ids]
    if not benign_sds:
        return 0.0
    mean_sd = sd_mean(benign_sds)
    se = 0.0
    for sd in benign_sds:
        se += sd_distance(sd, mean_sd) ** 2
    return float(se / len(benign_sds))

def save_plot(xs, ys, title, ylabel, fname):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# ---------------------------
# APTOS Dataset class

class APTOSDataset(data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------
# Simulation runner
# ---------------------------
def simulate(
    num_clients: int = 10,
    rounds: int = 30,
    num_malicious: int = 2,
    attack_type: str = "sign_flip",
    agg: str = "balance",
    seed: int = 42,
    trim_frac: float = 0.2,
    krum_f: int = 1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Replace MNIST with APTOS ----
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    csv_path = "/home/priyanka/flwr_app/my-awesome-app/data/train.csv"
    img_dir = "/home/priyanka/flwr_app/my-awesome-app/data/train_images"

    full_dataset = APTOSDataset(csv_path, img_dir, transform=transform)

    # Limit dataset size if debug_size is set
    if args.debug_size is not None:
        full_dataset = torch.utils.data.Subset(full_dataset, list(range(min(args.debug_size, len(full_dataset)))))


    # Split into train/test (80/20)
    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    trainset_full, testset_full = torch.utils.data.random_split(full_dataset, [train_len, test_len])


    # split
    train_len = len(trainset_full) // num_clients
    test_len = len(testset_full) // num_clients
    train_splits = torch.utils.data.random_split(trainset_full, [train_len] * (num_clients - 1) + [len(trainset_full) - train_len * (num_clients - 1)])
    test_splits = torch.utils.data.random_split(testset_full, [test_len] * (num_clients - 1) + [len(testset_full) - test_len * (num_clients - 1)])

    # graph
    k = min(4, num_clients - 1)
    p = 0.2
    G = nx.watts_strogatz_graph(n=num_clients, k=k, p=p, seed=seed)

    malicious_ids = set(random.sample(range(num_clients), k=num_malicious)) if num_malicious > 0 else set()

    clients: List[BalanceClient] = []
    for i in range(num_clients):
        is_mal = i in malicious_ids
        client = BalanceClient(
            cid=i,
            trainset=train_splits[i],
            testset=test_splits[i],
            neighbors=list(G.neighbors(i)),
            alpha=0.5,
            gamma=0.3,
            kappa=1.0,
            rounds=rounds,
            is_malicious=is_mal,
            attack_type=attack_type if is_mal else "none",
            device=device,
        )
        clients.append(client)

    # backdoor test dataset
    backdoor_xs, backdoor_ys = make_backdoor_testset(testset_full, trigger_size=3, target_label=0, n_samples=200)

    # metrics collectors
    rounds_list = []
    max_ter_list = []
    max_asr_list = []
    consensus_list = []
    avg_benign_list = []
    avg_all_list = []

    print(f"Start sim: clients={num_clients}, malicious={len(malicious_ids)}, attack={attack_type}, agg={agg}")
    print("Malicious ids:", sorted(list(malicious_ids)))
    asr_history = {}
    import os, datetime

    # --- Create timestamped results folder ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{agg.upper()}_clients{num_clients}_mal{num_malicious}_{attack_type}"
    base_dir = os.path.join("results", f"{timestamp}_{run_name}")
    os.makedirs(base_dir, exist_ok=True)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for t in range(rounds):
        # local train
        intermediate = {}
        for c in clients:
            w = c.local_train(epochs=1)
            sent = c.craft_malicious_message(w)
            intermediate[c.cid] = sent
        
        
        # for each client, prepare neighbor messages
        for c in clients:
            recv = {nid: intermediate[nid] for nid in c.neighbors}
            new_w = None
            if agg == "balance":
                new_w = c.balance_aggregate(recv, t)
            elif agg == "fedavg":
                agg_sd = fedavg_agg(recv)
                new_w = agg_sd if agg_sd else c.model.state_dict()
            elif agg == "trim":
                agg_sd = trimmed_mean_agg(recv, trim_frac=trim_frac)
                new_w = agg_sd if agg_sd else c.model.state_dict()
            elif agg == "median":
                agg_sd = median_agg(recv)
                new_w = agg_sd if agg_sd else c.model.state_dict()
            elif agg == "krum":
                agg_sd = krum_agg(recv, f_est=krum_f)
                new_w = agg_sd if agg_sd else c.model.state_dict()
            else:
                new_w = c.balance_aggregate(recv, t)
            c.model.load_state_dict(new_w)

        # metrics
        max_ter = compute_max_ter(clients, malicious_ids)
        """max_asr = compute_max_asr(clients, malicious_ids, (backdoor_xs, backdoor_ys))"""

        # compute per-client ASR and store
        asr_per_clients = compute_asr_per_client(clients, malicious_ids, (backdoor_xs, backdoor_ys))
        max_asr = max(val for _, val in asr_per_clients) if asr_per_clients else 0.0

        for cid, val in asr_per_clients:
            if cid not in asr_history:
                asr_history[cid] = []
            asr_history[cid].append(val)


        consensus = compute_consensus_error(clients, malicious_ids)
        accs = [c.test_accuracy() for c in clients]
        benign_accs = [accs[i] for i in range(num_clients) if i not in malicious_ids]
        avg_all = float(np.mean(accs))
        avg_benign = float(np.mean(benign_accs)) if benign_accs else 0.0

        rounds_list.append(t + 1)
        max_ter_list.append(max_ter)
        max_asr_list.append(max_asr)
        consensus_list.append(consensus)
        avg_benign_list.append(avg_benign)
        avg_all_list.append(avg_all)

        if (t + 1) % max(1, rounds // 10) == 0 or t == rounds - 1:
            print(f"Round {t+1}/{rounds} - Max.TER={max_ter:.4f} Max.ASR={max_asr:.4f} Consensus={consensus:.4f} AvgBenign={avg_benign:.4f}")
    
    # Save each client's final trained model
    """for i, c in enumerate(clients):
        torch.save(c.model.state_dict(), f"client_{i}_final_model.pth")
    print("Saved all client models to .pth files")
    print("Simulation complete.")"""

    save_asr_curves(asr_history, rounds_list, output_dir=plots_dir)

    # also export per-client ASR CSV
    asr_csv = os.path.join(base_dir, f"asr_per_client_{agg}_clients{num_clients}_atk{attack_type}.csv")
    with open(asr_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["round"] + [f"client_{cid}" for cid in sorted(asr_history.keys())]
        writer.writerow(header)
        for ridx, r in enumerate(rounds_list):
            row = [r] + [asr_history[cid][ridx] for cid in sorted(asr_history.keys())]
            writer.writerow(row)
    print(f"Saved ASR per-client curves and CSV → {asr_csv}")

    # save metrics CSV
    fname = os.path.join(base_dir, f"metrics_{agg}_clients{num_clients}_mal{num_malicious}_atk{attack_type}.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "max_ter", "max_asr", "consensus", "avg_benign", "avg_all"])
        for r, a, b, c_, d, e in zip(rounds_list, max_ter_list, max_asr_list, consensus_list, avg_benign_list, avg_all_list):
            writer.writerow([r, a, b, c_, d, e])
    print(f"Saved metrics → {fname}")

    # save plots
    save_plot(rounds_list, max_ter_list, "Max Test Error Rate", "Max TER", os.path.join(plots_dir, f"max_ter_{agg}.png"))
    save_plot(rounds_list, max_asr_list, "Max Attack Success Rate", "Max ASR", os.path.join(plots_dir, f"max_asr_{agg}.png"))
    save_plot(rounds_list, consensus_list, "Consensus Error (Benign)", "Consensus Error", os.path.join(plots_dir, f"consensus_{agg}.png"))
    save_plot(rounds_list, avg_benign_list, "Average Benign Accuracy", "Accuracy", os.path.join(plots_dir, f"avg_benign_{agg}.png"))
    save_plot(rounds_list, avg_all_list, "Average All Clients Accuracy", "Accuracy", os.path.join(plots_dir, f"avg_all_{agg}.png"))
    print(f"Saved plots → {plots_dir}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--malicious", type=int, default=2)
    parser.add_argument("--attack", type=str, default="sign_flip", choices=["none", "random_noise", "sign_flip", "label_flip"])
    parser.add_argument("--agg", type=str, default="balance", choices=["balance", "fedavg", "trim", "median", "krum"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trim_frac", type=float, default=0.2)
    parser.add_argument("--krum_f", type=int, default=1)
    parser.add_argument("--debug_size", type=int, default=None, help="Use only a subset of dataset for debugging")
    args = parser.parse_args()

    simulate(
        num_clients=args.num_clients,
        rounds=args.rounds,
        num_malicious=args.malicious,
        attack_type=args.attack,
        agg=args.agg,
        seed=args.seed,
        trim_frac=args.trim_frac,
        krum_f=args.krum_f,
    )
