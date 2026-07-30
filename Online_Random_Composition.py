import random
import numpy as np
import torch
from torch.utils.data import Dataset


def last_nonzero_plus1(x: np.ndarray):
    """Find the last non-zero element index + 1 (active region length)."""
    if x.ndim == 1:
        nz = np.nonzero(x)[0]
    elif x.ndim == 2:
        nz = np.nonzero(np.any(x != 0, axis=0))[0]
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")
    return int(nz[-1] + 1) if len(nz) > 0 else 1


def as_2d_ct(x: np.ndarray, in_channels: int):
    """
    Convert input to 2D shape (channels, time) with exactly `in_channels` channels.
    If fewer channels, repeat the last channel; if more, truncate.
    """
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
    # repeat last channel to match required channels
    rep = [x[-1:]] * (in_channels - x.shape[0])
    return np.concatenate([x] + rep, axis=0)


class Online_Random_Composition(Dataset):
    """
    Online Random Composition dataset (Section III-B of the paper).
    Dynamically synthesizes multi-tab samples from single-tab homepage and subpage traces.
    """
    def __init__(
        self,
        train_home_npz: str,
        sub_with_home_npz: str,
        out_len: int = 20000,
        in_channels: int = 1,
        samples_per_epoch: int = 10000,
        num_tokens_choices: tuple = (3, 4, 5, 6),
        num_tokens_probs: tuple = (0.25, 0.35, 0.25, 0.15),
        p_home: float = 0.45,
        p_sub: float = 0.55,
        mmap: bool = True,
    ):
        super().__init__()
        # Load homepage data
        home = np.load(train_home_npz, mmap_mode=("r" if mmap else None))
        sub = np.load(sub_with_home_npz, mmap_mode=("r" if mmap else None))

        self.Xh = home["X"]
        self.yh = home["y"].astype(np.float32)

        self.Xs = sub["X"]
        self.ys_home = sub["y_home"].astype(np.float32)
        self.ys_sub = sub["y_sub"].astype(np.float32)

        self.num_main = self.yh.shape[1]
        self.num_sub = self.ys_sub.shape[1]

        self.out_len = int(out_len)
        self.in_channels = int(in_channels)
        self.samples_per_epoch = int(samples_per_epoch)

        self.num_tokens_choices = np.array(num_tokens_choices, dtype=np.int64)
        self.num_tokens_probs = np.array(num_tokens_probs, dtype=np.float64)
        self.num_tokens_probs = self.num_tokens_probs / self.num_tokens_probs.sum()

        s = float(p_home) + float(p_sub)
        self.p_home = float(p_home) / s
        self.p_sub = float(p_sub) / s

        # Precompute main class indices for fast sampling
        self.home_main_of_idx = np.argmax(self.yh, axis=1)
        self.sub_main_of_idx = np.argmax(self.ys_home, axis=1)
        self.sub_cls_of_idx = np.argmax(self.ys_sub, axis=1)

        # Group homepage indices by main class
        self.home_by_main: list = []
        for m in range(self.num_main):
            idxs = np.where(self.home_main_of_idx == m)[0]
            if len(idxs) == 0:
                raise ValueError(f"home main={m} has 0 samples")
            self.home_by_main.append(idxs)

        self.all_sub_idxs = np.arange(self.Xs.shape[0])

    def __len__(self):
        return self.samples_per_epoch

    def _pick_home(self, used_home_ids: set, used_home_mains: set):
        """Randomly pick a homepage trace whose main class is not used yet."""
        mains = np.random.permutation(self.num_main)
        for m in mains:
            if m in used_home_mains:
                continue
            cands = self.home_by_main[m]
            if len(used_home_ids) > 0:
                cands = cands[~np.isin(cands, list(used_home_ids))]
            if len(cands) == 0:
                continue
            idx = int(np.random.choice(cands))
            used_home_ids.add(idx)
            used_home_mains.add(m)
            return idx, int(m)
        return None, None

    def _pick_sub(self, used_sub_ids: set, used_sub_classes: set):
        """Randomly pick a subpage trace whose sub-class is not used yet."""
        cands = np.random.permutation(self.all_sub_idxs)
        for idx in cands:
            idx = int(idx)
            if idx in used_sub_ids:
                continue
            sc = int(self.sub_cls_of_idx[idx])
            if sc in used_sub_classes:
                continue
            used_sub_ids.add(idx)
            used_sub_classes.add(sc)
            m = int(self.sub_main_of_idx[idx])
            return idx, m, sc
        return None, None, None

    def __getitem__(self, idx):
        # Sample number of active websites (tabs)
        num_tokens = int(np.random.choice(self.num_tokens_choices, p=self.num_tokens_probs))
        out = np.zeros((1, self.out_len), dtype=np.float32)   # single-channel output
        y_main = np.zeros((self.num_main,), dtype=np.float32)

        used_home_ids = set()
        used_sub_ids = set()
        used_home_mains = set()
        used_sub_classes = set()

        segments = []
        success = 0
        trials = 0
        max_trials = max(20, num_tokens * 12)

        while success < num_tokens and trials < max_trials:
            trials += 1
            typ = 0 if np.random.rand() < self.p_home else 1

            if typ == 0:  # pick homepage
                h_idx, m = self._pick_home(used_home_ids, used_home_mains)
                if h_idx is None:
                    continue
                x = as_2d_ct(self.Xh[h_idx], self.in_channels)[:1, :]  # force 1 channel
                x = x[:, :last_nonzero_plus1(x)]   # crop to active region
                segments.append(x)
                y_main[m] = 1.0
                success += 1
            else:  # pick subpage
                s_idx, m, _ = self._pick_sub(used_sub_ids, used_sub_classes)
                if s_idx is None:
                    continue
                x = as_2d_ct(self.Xs[s_idx], self.in_channels)[:1, :]
                x = x[:, :last_nonzero_plus1(x)]
                segments.append(x)
                y_main[m] = 1.0
                success += 1

        # Fallback: if no segment was picked (should rarely happen)
        if len(segments) == 0:
            s_idx = int(np.random.randint(0, self.Xs.shape[0]))
            m = int(self.sub_main_of_idx[s_idx])
            x = as_2d_ct(self.Xs[s_idx], self.in_channels)[:1, :]
            x = x[:, :last_nonzero_plus1(x)]
            segments.append(x)
            y_main[m] = 1.0

        # Randomly shuffle the order of segments (traffic interleaving)
        random.shuffle(segments)

        # Concatenate into the fixed-length output
        pos = 0
        for seg in segments:
            remain = self.out_len - pos
            if remain <= 0:
                break
            seg_len = seg.shape[-1]
            if seg_len > remain:
                # Randomly crop a sub-window if segment exceeds remaining space
                start = np.random.randint(0, seg_len - remain + 1)
                seg = seg[:, start:start + remain]
                seg_len = seg.shape[-1]
            out[:, pos:pos + seg_len] = seg
            pos += seg_len

        return torch.from_numpy(out), torch.from_numpy(y_main)