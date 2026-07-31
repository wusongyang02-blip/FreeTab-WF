"""
Fine-tuning script for FreeTab-WF (Plan A: full parameter fine-tuning with low learning rate).
Adapts a pre-trained model using a small set of real multi-tab training traces.
"""

import os
import random
import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from FreeTab_WF import FreeTab_WF, FreeTabWFConfig


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
# Loss
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


# ================================
# Dataset
# ================================
class FixedLenMainOnlyDataset(Dataset):
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
            raise KeyError(f"Cannot find label in {npz_path}. keys={list(data.keys())}")
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
# Main: Fine-tuning (Plan A)
# ================================
def run_finetune_planA():
    # ---------- Paths (adjust to your own paths) ----------
    pretrained_ckpt = "./runs_FreeTab/best.pt"
    train_npz = "./Datasets/Chrome/Fine_tuning_Train.npz"
    test_2tab = "./Datasets/Chrome/Fine_tuning_2tab_Test.npz"
    test_3tab = "./Datasets/Chrome/Fine_tuning_3tab_Test.npz"
    save_dir = "./runs_finetune_planA"
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Hyperparameters ----------
    in_channels = 1
    prefix_len = 20000
    batch_size = 32
    epochs = 20
    lr = 1e-5                  # Plan A: very low learning rate
    weight_decay = 1e-3
    grad_clip = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    set_seed(127)

    # ---------- Load pre-trained model ----------
    print(f"Loading pre-trained FreeTab-WF from {pretrained_ckpt}")
    ckpt = torch.load(pretrained_ckpt, map_location='cpu')
    cfg = FreeTabWFConfig(**ckpt['cfg'])
    model = FreeTab_WF(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    print("Pre-trained weights loaded.")

    # Enable full parameter fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,} (all)")

    # ---------- Load fine-tuning training set ----------
    train_ds = FixedLenMainOnlyDataset(
        train_npz, prefix_len, in_channels,
        x_key="X", y_key_candidates=("y_home", "y_main"), mmap=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    num_main = train_ds.y.shape[1]
    print(f"Number of monitored websites: {num_main}")

    # ---------- Load test sets ----------
    test_2tab_ds = FixedLenMainOnlyDataset(
        test_2tab, prefix_len, in_channels,
        x_key="X", y_key_candidates=("y_home", "y_main"), mmap=True
    )
    test_3tab_ds = FixedLenMainOnlyDataset(
        test_3tab, prefix_len, in_channels,
        x_key="X", y_key_candidates=("y_home", "y_main"), mmap=True
    )
    test_loader_2tab = DataLoader(test_2tab_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)
    test_loader_3tab = DataLoader(test_3tab_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)

    # ---------- Optimizer & Loss ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = AsymmetricLoss(gamma_neg=2.0, gamma_pos=1.0, clip=0.0)

    # ---------- Fine-tuning loop ----------
    print("Start fine-tuning (Plan A: lr=1e-5, epochs=20, full parameter)...")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs} | loss {train_loss:.6f} | time {dt:.1f}s")
        if ep % 5 == 0 or ep == epochs:
            torch.save({
                'model': model.state_dict(),
                'epoch': ep,
                'loss': train_loss,
                'cfg': cfg.__dict__,
            }, os.path.join(save_dir, f"checkpoint_ep{ep}.pt"))

    # Save final model
    final_ckpt = os.path.join(save_dir, "finetuned_final.pt")
    torch.save({'model': model.state_dict(), 'epoch': epochs, 'cfg': cfg.__dict__}, final_ckpt)
    print(f"Final model saved to {final_ckpt}")

    # ---------- Evaluation ----------
    threshold = ckpt.get('thr_main', 0.5)
    print(f"Using threshold: {threshold:.2f}")

    test_report_2tab = evaluate_main(model, test_loader_2tab, device, threshold)
    test_report_3tab = evaluate_main(model, test_loader_3tab, device, threshold)

    print("\n==== FINAL TEST (2-tab) ====")
    for k, v in test_report_2tab.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n==== FINAL TEST (3-tab) ====")
    for k, v in test_report_3tab.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Save results
    with open(os.path.join(save_dir, "final_test_metrics.txt"), "w") as f:
        f.write("==== FINAL TEST (2-tab) ====\n")
        for k, v in test_report_2tab.items():
            f.write(f"{k}: {v:.6f}\n" if isinstance(v, float) else f"{k}: {v}\n")
        f.write("\n==== FINAL TEST (3-tab) ====\n")
        for k, v in test_report_3tab.items():
            f.write(f"{k}: {v:.6f}\n" if isinstance(v, float) else f"{k}: {v}\n")
    print("Results saved to:", os.path.join(save_dir, "final_test_metrics.txt"))


if __name__ == "__main__":
    run_finetune_planA()