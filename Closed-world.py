"""
Training and evaluation script for FreeTab-WF.
This script uses Online_Random_Composition for training data generation
and the FreeTab-WF model for multi-label website fingerprinting.
"""

import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from FreeTab-WF import FreeTab_WF, FreeTabWFConfig
from Online_Random_Composition import Online_Random_Composition


# ================================
# Utilities
# ================================
def set_seed(seed: int = 127):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_2d_ct(x: np.ndarray, in_channels: int):
    """Convert input to 2D (channels, time) with exactly `in_channels`."""
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
# Evaluation Dataset (fixed length)
# ================================
class FixedLenMainOnlyDataset(Dataset):
    """Fixed-length dataset for validation and testing."""
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
        x = as_2d_ct(self.X[idx], self.in_channels)[:1, :]   # force single channel
        out = np.zeros((1, self.prefix_len), dtype=np.float32)
        L = min(self.prefix_len, x.shape[-1])
        out[:, :L] = x[:, :L]
        y = self.y[idx]
        return torch.from_numpy(out), torch.from_numpy(y)


# ================================
# Loss Function (Asymmetric Loss)
# ================================
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification."""
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


# ================================
# Metrics
# ================================
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
# Training / Evaluation helpers
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
# Efficiency measurements
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
# Main training routine
# ================================
def run():
    # ====== Data paths (please adjust to your own paths) ======
    train_home_npz = "./Datasets/Chrome/Chrome_Homepage_Single.npz"    # or Tor/Tor_Homepage_Single.npz
    sub_with_home_npz = "./Datasets/Chrome/Chrome_Subpage_Single.npz"    # or Tor/Tor_Subpage_Single.npz
    val_npz = "./Datasets/Chrome/Chrome_Val.npz"    # or Tor/Tor_Val.npz (Drift/Drift_Val.npz)
    test_npz = "./Datasets/Chrome/Chrome_Test.npz"    # or Tor/Tor_Test.npz (Drift/Drift_Val.npz)

    save_dir = "./runs_FreeTab"
    os.makedirs(save_dir, exist_ok=True)

    # ====== Setup ======
    seed = 127
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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
    thresholds = [i / 100 for i in range(20, 91, 2)]

    # ====== Datasets and DataLoaders ======
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

    val_ds = FixedLenMainOnlyDataset(
        npz_path=val_npz,
        prefix_len=prefix_len_eval,
        in_channels=in_channels,
        x_key="X",
        y_key_candidates=("y_home", "y_main"),
        mmap=True,
    )

    test_ds = FixedLenMainOnlyDataset(
        npz_path=test_npz,
        prefix_len=prefix_len_eval,
        in_channels=in_channels,
        x_key="X",
        y_key_candidates=("y_home", "y_main"),
        mmap=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    num_main = int(val_ds.y.shape[1])
    print("Number of monitored websites:", num_main)

    # ====== Model ======
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

    # ====== Training loop with validation ======
    best_val_f1 = -1.0
    best_ckpt = os.path.join(save_dir, "best.pt")

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

    # ====== Load best model and evaluate ======
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    thr = float(ckpt["thr_main"])

    print("\nLoaded best model from:", best_ckpt)
    print("Best validation main metrics:", ckpt["best_main"], "epoch:", ckpt["epoch"])

    val_report = evaluate_main(model, val_loader, device, thr)
    test_report = evaluate_main(model, test_loader, device, thr)

    print("\n==== FINAL VAL (MAIN) ====")
    for k, v in val_report.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n==== FINAL TEST (MAIN) ====")
    for k, v in test_report.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # ====== Efficiency ======
    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)
    latency_ms, extra = benchmark_inference(model, test_loader, device, warmup=30, iters=200)
    throughput, peak_mem_mb = (None, None) if extra is None else extra

    print("\n==== EFFICIENCY (LIGHTWEIGHT) ====")
    print(f"Params(total): {total_params}")
    print(f"Params(trainable): {trainable_params}")
    print(f"Model size (MB, params+buffers): {size_mb:.4f}")
    if latency_ms is not None:
        print(f"Inference latency (ms/sample): {latency_ms:.4f}")
    if throughput is not None:
        print(f"Throughput (samples/s): {throughput:.2f}")
    if peak_mem_mb is not None:
        print(f"Peak GPU memory (MB): {peak_mem_mb:.2f}")

    # ====== Save metrics to file ======
    out_txt = os.path.join(save_dir, "final_main_metrics.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("==== FINAL VAL (MAIN) ====\n")
        for k, v in val_report.items():
            f.write(f"{k}: {v:.6f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        f.write("\n==== FINAL TEST (MAIN) ====\n")
        for k, v in test_report.items():
            f.write(f"{k}: {v:.6f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        f.write("\n==== EFFICIENCY (LIGHTWEIGHT) ====\n")
        f.write(f"Params(total): {total_params}\n")
        f.write(f"Params(trainable): {trainable_params}\n")
        f.write(f"Model size (MB, params+buffers): {size_mb:.6f}\n")
        if latency_ms is not None:
            f.write(f"Inference latency (ms/sample): {latency_ms:.6f}\n")
        if throughput is not None:
            f.write(f"Throughput (samples/s): {throughput:.6f}\n")
        if peak_mem_mb is not None:
            f.write(f"Peak GPU memory (MB): {peak_mem_mb:.6f}\n")

    print("Saved metrics to:", out_txt)


if __name__ == "__main__":
    run()
