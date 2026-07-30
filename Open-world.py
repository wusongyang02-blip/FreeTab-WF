"""
Open-world training and evaluation script for FreeTab-WF.
Combines monitored and unmonitored traffic for robust evaluation.
"""

import os
import random
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from FreeTab-WF import FreeTab_WF, FreeTabWFConfig
from Online_Random_Composition import Online_Random_Composition


# ================================
# Utilities
# ================================
def set_seed(seed=127):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def last_nonzero_plus1(x: np.ndarray):
    if x.ndim == 1:
        nz = np.nonzero(x)[0]
    elif x.ndim == 2:
        nz = np.nonzero(np.any(x != 0, axis=0))[0]
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")
    return int(nz[-1] + 1) if len(nz) > 0 else 1


def as_2d_ct(x: np.ndarray, in_channels: int):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported x shape: {x.shape}")

    if x.shape[0] == in_channels:
        return x
    if x.shape[0] > in_channels:
        return x[:in_channels]
    rep = [x[-1:]] * (in_channels - x.shape[0])
    return np.concatenate([x] + rep, axis=0)


# ================================
# Loss & metrics
# ================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2.0, gamma_pos=1.0, clip=0.0, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal = torch.pow(1.0 - pt, gamma)
        loss = -loss * focal

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@torch.no_grad()
def multilabel_prf1_from_preds(preds01: torch.Tensor, targets01: torch.Tensor, eps=1e-9):
    p = preds01.to(torch.int32)
    t = targets01.to(torch.int32)
    tp = (p & t).sum().item()
    fp = (p & (1 - t)).sum().item()
    fn = ((1 - p) & t).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


@torch.no_grad()
def preds_threshold(logits_cpu: torch.Tensor, threshold: float):
    return (torch.sigmoid(logits_cpu) >= threshold).to(torch.int32)


@torch.no_grad()
def threshold_sweep(logits_cpu: torch.Tensor, targets_cpu: torch.Tensor, thresholds):
    best = {"threshold": None, "precision": -1, "recall": -1, "f1": -1}
    t = targets_cpu.to(torch.int32)
    for thr in thresholds:
        preds = preds_threshold(logits_cpu, float(thr))
        p, r, f1 = multilabel_prf1_from_preds(preds, t)
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "precision": float(p), "recall": float(r), "f1": float(f1)}
    return best


# ================================
# Open-World metrics (per-label binary classification)
# ================================
def binary_precision_recall_f1(tp: int, fp: int, fn: int, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


@torch.no_grad()
def evaluate_open_world_per_label(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    num_classes: int,
):
    """
    Per-label binary evaluation for open-world setting.
    For each website class, positive = samples containing that website,
    negative = all other samples (including other monitored and open-world).
    Returns per-label metrics and macro-averaged results.
    """
    model.eval()

    all_logits = []
    all_labels = []

    for x, y_multi in loader:
        x = x.to(device, non_blocking=True)
        logits, _ = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_multi.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits)

    per_label_results = []

    for c in range(num_classes):
        y_true = labels[:, c].int().numpy()
        y_pred = (probs[:, c] >= threshold).int().numpy()
        y_score = probs[:, c].numpy()

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision, recall, f1 = binary_precision_recall_f1(tp, fp, fn)

        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.5

        per_label_results.append({
            "class": c,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    avg_precision = np.mean([r["precision"] for r in per_label_results])
    avg_recall = np.mean([r["recall"] for r in per_label_results])
    avg_f1 = np.mean([r["f1"] for r in per_label_results])
    avg_auc = np.mean([r["auc"] for r in per_label_results])

    return {
        "per_label": per_label_results,
        "macro_precision": avg_precision,
        "macro_recall": avg_recall,
        "macro_f1": avg_f1,
        "macro_auc": avg_auc,
    }


# ================================
# Datasets
# ================================
class FixedLenMainOnlyDataset(Dataset):
    """Fixed-length dataset for validation and testing (monitored only)."""
    def __init__(
        self,
        npz_path: str,
        prefix_len: int,
        in_channels: int = 1,
        x_key: str = "X",
        y_key_candidates: Tuple[str, ...] = ("y_home", "y_main"),
        mmap: bool = True,
    ):
        data = np.load(npz_path, mmap_mode=("r" if mmap else None))
        self.X = data[x_key]

        y_key = None
        for k in y_key_candidates:
            if k in data:
                y_key = k
                break
        if y_key is None:
            raise KeyError(f"Cannot find y_home/y_main in {npz_path}")

        self.y = data[y_key].astype(np.float32)
        self.prefix_len = int(prefix_len)
        self.in_channels = int(in_channels)

        assert self.X.shape[0] == self.y.shape[0]
        assert self.y.ndim == 2

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = as_2d_ct(self.X[idx], self.in_channels)[:1, :]
        out = np.zeros((1, self.prefix_len), dtype=np.float32)
        L = min(self.prefix_len, x.shape[-1])
        out[:, :L] = x[:, :L]
        y = self.y[idx]
        return torch.from_numpy(out), torch.from_numpy(y)


class OpenWorldDataset(Dataset):
    """Open-world dataset: all samples are unmonitored (labels are all zeros)."""
    def __init__(
        self,
        npz_path: str,
        prefix_len: int,
        num_classes: int,
        in_channels: int = 1,
        x_key: str = "X",
        y_key: str = "y_main",
        mmap: bool = True,
        sample_limit: Optional[int] = None,
    ):
        data = np.load(npz_path, mmap_mode=("r" if mmap else None))
        self.X = data[x_key]

        # Check y_key (should be all zeros)
        if y_key in data:
            y = data[y_key]
            if np.any(y != 0):
                print(f"Warning: open-world {y_key} has non-zero entries! sum={y.sum()}, shape={y.shape}")
            else:
                print(f"Open-world {y_key} is all zeros (expected). shape={y.shape}")
        else:
            print(f"Warning: no {y_key} found in open-world npz, assuming all zeros")

        self.num_classes = num_classes
        self.prefix_len = int(prefix_len)
        self.in_channels = int(in_channels)

        self.num_samples = self.X.shape[0]
        if sample_limit is not None and sample_limit < self.num_samples:
            indices = np.random.choice(self.num_samples, sample_limit, replace=False)
            self.X = self.X[indices]
            self.num_samples = len(indices)
            print(f"Open-world sampled {self.num_samples} samples")

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        x = as_2d_ct(self.X[idx], self.in_channels)[:1, :]
        out = np.zeros((1, self.prefix_len), dtype=np.float32)
        L = min(self.prefix_len, x.shape[-1])
        out[:, :L] = x[:, :L]
        y = np.zeros(self.num_classes, dtype=np.float32)
        return torch.from_numpy(out), torch.from_numpy(y)


class OpenWorldMixedDataset(Dataset):
    """Mix monitored and open-world samples for test evaluation."""
    def __init__(
        self,
        monitored_dataset: FixedLenMainOnlyDataset,
        open_world_dataset: OpenWorldDataset,
        monitored_ratio: float = 0.5,
        total_samples: int = None,
        seed: int = 42,
    ):
        self.monitored = monitored_dataset
        self.open_world = open_world_dataset

        n_monitored = len(self.monitored)
        n_open = len(self.open_world)

        if total_samples is None:
            total_samples = n_monitored + n_open

        n_mon_use = int(total_samples * monitored_ratio)
        n_mon_use = min(n_mon_use, n_monitored)
        n_open_use = total_samples - n_mon_use
        n_open_use = min(n_open_use, n_open)

        if n_mon_use + n_open_use < total_samples:
            if n_mon_use < n_monitored:
                n_mon_use = min(n_monitored, total_samples - n_open_use)
            if n_open_use < n_open:
                n_open_use = min(n_open, total_samples - n_mon_use)
            total_samples = n_mon_use + n_open_use

        print(f"Mixed dataset: {n_mon_use} monitored + {n_open_use} open-world = {total_samples} samples")

        np.random.seed(seed)
        if n_mon_use < n_monitored:
            mon_indices = np.random.choice(n_monitored, n_mon_use, replace=False)
        else:
            mon_indices = np.arange(n_monitored)
        self.mon_indices = mon_indices

        if n_open_use < n_open:
            open_indices = np.random.choice(n_open, n_open_use, replace=False)
        else:
            open_indices = np.arange(n_open)
        self.open_indices = open_indices

        self.samples = []
        for idx in self.mon_indices:
            self.samples.append(("mon", int(idx)))
        for idx in self.open_indices:
            self.samples.append(("open", int(idx)))

        np.random.seed(seed)
        perm = np.random.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        typ, idx2 = self.samples[idx]
        if typ == "mon":
            x, y_multi = self.monitored[idx2]
            return x, y_multi
        else:
            x, y_multi = self.open_world[idx2]
            return x, y_multi


# ================================
# Train / Eval helpers
# ================================
@torch.no_grad()
def collect_outputs_main(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _ = model(x)
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        total += float(loss.detach().cpu().item()) * bs
        n += bs
    return total / max(n, 1)


def evaluate_main(model, loader, device, threshold):
    logits, y = collect_outputs_main(model, loader, device)
    pred = preds_threshold(logits, threshold)
    p, r, f1 = multilabel_prf1_from_preds(pred, y.to(torch.int32))
    return {"thr": float(threshold), "precision": float(p), "recall": float(r), "f1": float(f1)}


# ================================
# Efficiency metrics
# ================================
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 ** 2)


@torch.no_grad()
def benchmark_inference(model: nn.Module, loader: DataLoader, device, warmup=30, iters=200):
    model.eval()
    xs = []
    need = warmup + iters
    for x, _ in loader:
        xs.append(x.to(device, non_blocking=True))
        if sum(t.shape[0] for t in xs) >= need:
            break
    if len(xs) == 0:
        return None, None

    xcat = torch.cat(xs, dim=0)[:need]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for i in range(warmup):
        _ = model(xcat[i:i+1])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    for i in range(iters):
        _ = model(xcat[warmup + i:warmup + i + 1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.time()

    latency_ms = (t1 - t0) * 1000.0 / iters
    throughput = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return latency_ms, (throughput, peak_mem_mb)


# ================================
# Main
# ================================
def run():
    # ---------- Data paths (adjust to your own paths) ----------
    train_home_npz = "./Datasets/Chrome_Homepage_Single.npz"
    sub_with_home_npz = "./data/Chrome_Subpage_Single.npz"
    val_npz = "./Datasets/Chrome_Val.npz"
    test_npz = "./Datasets/Chrome_Test.npz"
    open_world_npz = "./Datasets/Open_world.npz"

    save_dir = "./runs_FreeTab_openworld"
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Hyperparameters ----------
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    in_channels = 1
    out_len_train = 20000
    prefix_len_eval = 20000

    epochs = 100
    batch_size = 32
    num_workers = 0
    lr = 1e-3
    weight_decay = 1e-3
    grad_clip = 1.0
    samples_per_epoch_train = 12000
    thresholds = [i / 100 for i in range(20, 71, 2)]

    # ---------- Training data (Online Random Composition) ----------
    train_ds = Online_Random_Composition(
        train_home_npz=train_home_npz,
        sub_with_home_npz=sub_with_home_npz,
        out_len=out_len_train,
        in_channels=in_channels,
        samples_per_epoch=samples_per_epoch_train,
        num_tokens_choices=(5, 6),
        num_tokens_probs=(0.5, 0.5),
        p_home=0.45,
        p_sub=0.55,
        mmap=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    # ---------- Validation data ----------
    val_ds = FixedLenMainOnlyDataset(
        npz_path=val_npz,
        prefix_len=prefix_len_eval,
        in_channels=in_channels,
        x_key="X",
        y_key_candidates=("y_home", "y_main"),
        mmap=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ---------- Model (FreeTab-WF) ----------
    num_main = int(val_ds.y.shape[1])
    print("num_main:", num_main)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = AsymmetricLoss(gamma_neg=2.0, gamma_pos=1.0, clip=0.0)

    best_val_f1 = -1.0
    best_ckpt = os.path.join(save_dir, "best.pt")

    # ---------- Training loop ----------
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)

        val_logits, val_y = collect_outputs_main(model, val_loader, device)
        best_main = threshold_sweep(val_logits, val_y, thresholds)

        if best_main["f1"] > best_val_f1:
            best_val_f1 = best_main["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "thr_main": float(best_main["threshold"]),
                    "best_main": best_main,
                    "epoch": ep,
                    "val_f1": float(best_val_f1),
                    "cfg": cfg.__dict__,
                },
                best_ckpt,
            )

        dt = time.time() - t0
        print(
            f"Epoch {ep:03d}/{epochs} | "
            f"loss {train_loss:.6f} | "
            f"VAL thr={best_main['threshold']:.2f} "
            f"P={best_main['precision']:.4f} R={best_main['recall']:.4f} F1={best_main['f1']:.4f} | "
            f"time {dt:.1f}s"
        )

    # ---------- Load best model ----------
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    thr = float(ckpt["thr_main"])

    print("\nLoaded best:", best_ckpt)
    print("best_main:", ckpt["best_main"], "epoch:", ckpt["epoch"])

    # ---------- Closed-World test (monitored only) ----------
    test_mon_ds = FixedLenMainOnlyDataset(
        npz_path=test_npz,
        prefix_len=prefix_len_eval,
        in_channels=in_channels,
        x_key="X",
        y_key_candidates=("y_home", "y_main"),
        mmap=True,
    )
    test_mon_loader = DataLoader(
        test_mon_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_report = evaluate_main(model, test_mon_loader, device, thr)

    print("\n==== FINAL TEST (Closed-World, MONITORED ONLY) ====")
    print(f"Precision: {test_report['precision']:.6f}")
    print(f"Recall: {test_report['recall']:.6f}")
    print(f"F1: {test_report['f1']:.6f}")

    # ---------- Open-World evaluation ----------
    open_ds = OpenWorldDataset(
        npz_path=open_world_npz,
        prefix_len=prefix_len_eval,
        num_classes=num_main,
        in_channels=in_channels,
        x_key="X",
        y_key="y_main",
        mmap=True,
        sample_limit=4000,
    )

    mixed_test_ds = OpenWorldMixedDataset(
        monitored_dataset=test_mon_ds,
        open_world_dataset=open_ds,
        monitored_ratio=1.0 / 3.0,
        total_samples=3000,
        seed=seed,
    )
    test_loader = DataLoader(
        mixed_test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    open_results = evaluate_open_world_per_label(
        model, test_loader, device, thr, num_main
    )

    print("\n==== OPEN-WORLD EVALUATION (Per-Label, Macro-Averaged) ====")
    print(f"Macro Precision: {open_results['macro_precision']:.6f}")
    print(f"Macro Recall: {open_results['macro_recall']:.6f}")
    print(f"Macro F1: {open_results['macro_f1']:.6f}")
    print(f"Macro AUC: {open_results['macro_auc']:.6f}")

    # ---------- Efficiency metrics ----------
    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)
    latency_ms, extra = benchmark_inference(model, test_loader, device, warmup=30, iters=200)
    throughput, peak_mem_mb = (None, None) if extra is None else extra

    print("\n==== EFFICIENCY (LIGHTWEIGHT) ====")
    print(f"Params(total): {total_params}")
    print(f"Params(trainable): {trainable_params}")
    print(f"Model size (MB): {size_mb:.4f}")
    if latency_ms is not None:
        print(f"Inference latency (ms/sample): {latency_ms:.4f}")
    if throughput is not None:
        print(f"Throughput (samples/s): {throughput:.2f}")
    if peak_mem_mb is not None:
        print(f"Peak GPU memory (MB): {peak_mem_mb:.2f}")

    # ---------- Save results ----------
    out_txt = os.path.join(save_dir, "open_world_metrics.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("==== FINAL TEST (Closed-World, MONITORED ONLY) ====\n")
        f.write(f"precision: {test_report['precision']:.6f}\n")
        f.write(f"recall: {test_report['recall']:.6f}\n")
        f.write(f"f1: {test_report['f1']:.6f}\n")

        f.write("\n==== OPEN-WORLD EVALUATION (Per-Label, Macro-Averaged) ====\n")
        f.write(f"macro_precision: {open_results['macro_precision']:.6f}\n")
        f.write(f"macro_recall: {open_results['macro_recall']:.6f}\n")
        f.write(f"macro_f1: {open_results['macro_f1']:.6f}\n")
        f.write(f"macro_auc: {open_results['macro_auc']:.6f}\n")

        f.write("\n==== PER-LABEL BREAKDOWN ====\n")
        for r in open_results['per_label']:
            f.write(f"Class {r['class']:3d}: P={r['precision']:.6f} R={r['recall']:.6f} F1={r['f1']:.6f} AUC={r['auc']:.6f}\n")

        f.write("\n==== EFFICIENCY ====\n")
        f.write(f"Params(total): {total_params}\n")
        f.write(f"Params(trainable): {trainable_params}\n")
        f.write(f"Model size (MB): {size_mb:.6f}\n")
        if latency_ms is not None:
            f.write(f"Inference latency (ms/sample): {latency_ms:.6f}\n")
        if throughput is not None:
            f.write(f"Throughput (samples/s): {throughput:.6f}\n")
        if peak_mem_mb is not None:
            f.write(f"Peak GPU memory (MB): {peak_mem_mb:.6f}\n")

    print(f"\nSaved metrics to: {out_txt}")


if __name__ == "__main__":
    run()
