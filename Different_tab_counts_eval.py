"""
Evaluation script for FreeTab-WF on multi-tab closed-world test sets.
Evaluates per-tab-count performance (2-tab, 3-tab, 4-tab, 5-tab) with P@K and MAP@K.
"""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from FreeTab-WF import FreeTab_WF, FreeTabWFConfig


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_2d_ct(x: np.ndarray, in_channels: int):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"Unsupported x shape: {x.shape}")

    if x.shape[0] == in_channels:
        return x
    if x.shape[0] > in_channels:
        return x[:in_channels]
    rep = [x[-1:]] * (in_channels - x.shape[0])
    return np.concatenate([x] + rep, axis=0)


class FixedLenMainOnlyDataset(Dataset):
    """Fixed-length dataset for evaluation on multi-tab traffic."""
    def __init__(
        self,
        npz_path,
        prefix_len,
        in_channels=2,
        x_key="X",
        y_key_candidates=("y_home", "y_main"),
        mmap=True
    ):
        data = np.load(npz_path, mmap_mode=("r" if mmap else None))
        self.X = data[x_key]

        y_key = None
        for k in y_key_candidates:
            if k in data:
                y_key = k
                break
        if y_key is None:
            raise KeyError(f"Cannot find y_home/y_main in {npz_path}, keys={list(data.keys())}")

        self.y = data[y_key].astype(np.float32)
        self.prefix_len = int(prefix_len)
        self.in_channels = int(in_channels)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = as_2d_ct(self.X[idx], self.in_channels)
        out = np.zeros((self.in_channels, self.prefix_len), dtype=np.float32)
        L = min(self.prefix_len, x.shape[-1])
        out[:, :L] = x[:, :L]
        return torch.from_numpy(out), torch.from_numpy(self.y[idx])


@torch.no_grad()
def collect_outputs_main(model, loader, device):
    """Collect all logits and labels from the dataloader."""
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _ = model(x)
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


@torch.no_grad()
def preds_threshold(logits, thr):
    """Convert logits to binary predictions using a threshold."""
    return (torch.sigmoid(logits) >= thr).to(torch.int32)


@torch.no_grad()
def micro_prf1(pred01, target01, eps=1e-9):
    """Compute micro-averaged precision, recall, and F1."""
    p = pred01.to(torch.int32)
    t = target01.to(torch.int32)
    tp = (p & t).sum().item()
    fp = (p & (1 - t)).sum().item()
    fn = ((1 - p) & t).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def infer_fixed_k(targets: np.ndarray) -> int:
    """
    Infer the number of active websites per sample (K).
    Raises ValueError if K is not fixed across all samples.
    """
    per_sample_k = targets.sum(axis=1)
    uniq = np.unique(per_sample_k)
    if len(uniq) != 1:
        raise ValueError(
            f"Labels do not have a fixed K. Found values: {uniq.tolist()}. "
            "P@K and MAP@K require a fixed K."
        )
    k = int(uniq[0])
    if k <= 0:
        raise ValueError(f"Invalid K={k}")
    return k


@torch.no_grad()
def precision_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Compute Precision@K for multi-label classification."""
    N, C = scores.shape
    k = min(k, C)
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rel = np.take_along_axis(targets, topk, axis=1)
    return float((rel.sum(axis=1) / float(k)).mean())


@torch.no_grad()
def map_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Compute Mean Average Precision@K for multi-label classification."""
    N, C = scores.shape
    k = min(k, C)
    ap_list = []
    for i in range(N):
        s = scores[i]
        t = targets[i]
        idx = np.argsort(-s)[:k]
        rel = t[idx].astype(np.float32)

        total_rel = int(t.sum())
        if total_rel == 0:
            ap_list.append(0.0)
            continue

        denom = float(min(total_rel, k))
        cumsum = np.cumsum(rel)
        prec = cumsum / (np.arange(k) + 1.0)
        ap = float((prec * rel).sum() / denom)
        ap_list.append(ap)

    return float(np.mean(ap_list))


def evaluate_one(model, loader, device, thr):
    """
    Evaluate the model on a single test set.
    Returns precision, recall, F1, P@K, and MAP@K.
    """
    logits, y = collect_outputs_main(model, loader, device)
    pred = preds_threshold(logits, thr)
    p, r, f1 = micro_prf1(pred, y.to(torch.int32))

    probs = torch.sigmoid(logits).numpy()
    targets = y.numpy().astype(np.int32)

    k = infer_fixed_k(targets)
    p_at_k = precision_at_k(probs, targets, k)
    mapk = map_at_k(probs, targets, k)

    out = {
        "thr": float(thr),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "k": int(k),
        f"P@{k}": float(p_at_k),
        f"MAP@{k}": float(mapk),
    }
    return out


def run():
    set_seed(127)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------- Paths (adjust to your own paths) ----------
    ckpt_path = "./runs_freeTab/best.pt"
    prefix_len_eval = 20000
    in_channels = 1
    batch_size = 64
    num_workers = 0

    test_paths = {
        "2tab_test": "./Datasets/Chrome/Chrome_2tab_test.npz",    # or Tor/Tor_2tab_test.npz (Drift/Drift_2tab.npz)
        "3tab_test": "./Datasets/Chrome/Chrome_3tab_test.npz",    # or Tor/Tor_3tab_test.npz (Drift/Drift_3tab.npz)
        "4tab_test": "./Datasets/Chrome/Chrome_4tab_test.npz",    # or Tor/Tor_4tab_test.npz (Drift/Drift_4tab.npz)
        "5tab_test": "./Datasets/Chrome/Chrome_5tab_test.npz",    # or Tor/Tor_5tab_test.npz (Drift/Drift_5tab.npz)
    }

    # ---------- Load checkpoint ----------
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    thr = float(ckpt.get("thr_main", 0.5))

    num_main = 50  # number of monitored websites

    # Build configuration from checkpoint if available
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg_dict = ckpt["cfg"].copy()
        cfg_dict["num_main_classes"] = num_main
        cfg_dict["in_channels"] = in_channels
        cfg = FreeTabWFConfig(**cfg_dict)
    else:
        cfg = FreeTabWFConfig(
            num_main_classes=num_main,
            in_channels=in_channels,
            cnn_base=32,
            cnn_depth=3,
            cnn_dropout=0.1,
            window_size=64,
            window_stride=56,
            head_dim=96,
            head_dropout=0.2,
        )

    model = FreeTab_WF(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("Loaded checkpoint:", ckpt_path, "threshold:", thr)

    # ---------- Evaluate on each test set ----------
    for name, path in test_paths.items():
        ds = FixedLenMainOnlyDataset(path, prefix_len_eval, in_channels=in_channels, mmap=True)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        rep = evaluate_one(model, loader, device, thr)

        k = rep["k"]
        print(f"\n==== {name} ====")
        print(f"thr={rep['thr']:.4f} P={rep['precision']:.6f} R={rep['recall']:.6f} F1={rep['f1']:.6f}")
        print(f"P@{k}={rep[f'P@{k}']:.6f}  MAP@{k}={rep[f'MAP@{k}']:.6f}")


if __name__ == "__main__":
    run()
