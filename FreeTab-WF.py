import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def make_padding_mask(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Generate padding mask from input tensor."""
    if x.dim() == 2:
        return (x.abs() > eps)
    if x.dim() == 3:
        return (x.abs().max(dim=1).values > eps)
    raise ValueError(f"Unsupported shape: {x.shape}")


def masked_mean_pool_1d(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply masked average pooling along the temporal dimension."""
    m = mask.float().unsqueeze(1)
    feat = feat * m
    denom = m.sum(dim=-1).clamp_min(eps)
    return feat.sum(dim=-1) / denom


class Conv1dBlock(nn.Module):
    """Single 1D convolutional block with BatchNorm, GELU, and Dropout."""
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int, dropout: float):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class TemporalCNNEncoder(nn.Module):
    """
    Temporal convolutional encoder that compresses input sequences
    while preserving local traffic patterns.
    """
    def __init__(self, in_ch: int = 1, base: int = 64, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            out = base * (2 ** min(i, 2))
            layers.append(Conv1dBlock(ch, out, k=9, stride=2, dropout=dropout))
            layers.append(Conv1dBlock(out, out, k=5, stride=1, dropout=dropout))
            ch = out
        self.net = nn.Sequential(*layers)
        self.out_dim = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerPool(nn.Module):
    """
    Transformer encoder with a learnable [CLS] token for global sequence aggregation.
    """
    def __init__(self, dim: int = 192, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, h_win: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            h_win: Window-level features of shape (B, N, D)
            mask: Boolean mask indicating valid windows, shape (B, N)
        Returns:
            Global feature vector from [CLS] token, shape (B, D)
        """
        B, N, D = h_win.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h_all = torch.cat([cls_tokens, h_win], dim=1)

        pad_mask = torch.zeros(B, N + 1, dtype=torch.bool, device=h_win.device)
        if mask is not None:
            pad_mask[:, 1:] = ~mask

        h_trans = self.transformer(h_all, src_key_padding_mask=pad_mask)
        return h_trans[:, 0, :], None


@dataclass
class FreeTabWFConfig:
    """Configuration for the FreeTab-WF model."""
    num_main_classes: int = 50
    in_channels: int = 1

    cnn_base: int = 64
    cnn_depth: int = 4
    cnn_dropout: float = 0.1

    window_size: int = 48
    window_stride: int = 24

    head_dim: int = 160
    head_dropout: float = 0.2


class FreeTab_WF(nn.Module):
    """
    FreeTab-WF: Multi-tab website fingerprinting model.
    
    Architecture consists of:
    1. Temporal CNN encoder for local feature extraction
    2. Overlapping window-based feature summarization
    3. Transformer aggregator with [CLS] token
    4. Multi-label classification head
    """
    def __init__(self, cfg: FreeTabWFConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Temporal convolutional encoder
        self.encoder = TemporalCNNEncoder(
            in_ch=cfg.in_channels,
            base=cfg.cnn_base,
            depth=cfg.cnn_depth,
            dropout=cfg.cnn_dropout
        )
        D = self.encoder.out_dim

        # 2. Projection layer from CNN feature dimension to Transformer dimension
        self.win_proj = nn.Sequential(
            nn.Linear(D, cfg.head_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
        )

        # 3. Transformer aggregator with learnable [CLS] token
        self.transformer_pool = TransformerPool(
            dim=cfg.head_dim, num_heads=4, num_layers=2, dropout=0.1
        )

        # 4. Multi-label classification head
        self.cls_main = nn.Linear(cfg.head_dim, cfg.num_main_classes)

    def _downsample_mask_nearest(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Downsample padding mask using nearest-neighbor interpolation."""
        m = mask.float().unsqueeze(1)
        m2 = F.interpolate(m, size=target_len, mode="nearest")
        return (m2.squeeze(1) > 0.5)

    def _make_windows(self, feat: torch.Tensor, mask: torch.Tensor):
        """
        Partition feature map into overlapping windows and apply masked average pooling.
        
        Returns:
            h: Window-level features of shape (B, num_windows, D)
            m_win: Boolean mask of valid windows, shape (B, num_windows)
        """
        B, D, T = feat.shape
        ws, st = self.cfg.window_size, self.cfg.window_stride

        # Pad if feature length is shorter than window size
        if T < ws:
            pad = ws - T
            feat = F.pad(feat, (0, pad))
            mask = F.pad(mask, (0, pad), value=False)

        # Unfold into overlapping windows
        feat_u = feat.unfold(dimension=2, size=ws, step=st)
        mask_u = mask.unfold(dimension=1, size=ws, step=st)
        m_win = mask_u.any(dim=-1)  # window is valid if it contains any valid signal

        N = feat_u.shape[2]
        feat_u = feat_u.permute(0, 2, 1, 3).contiguous().view(B * N, D, ws)
        mask_u = mask_u.contiguous().view(B * N, ws)

        # Masked average pooling per window
        h = masked_mean_pool_1d(feat_u, mask_u).view(B, N, D)
        return h, m_win

    def forward(self, x: torch.Tensor):
        """
        Forward pass of FreeTab-WF.
        
        Args:
            x: Input traffic trace of shape (B, L) or (B, 1, L)
            
        Returns:
            main_logits: Multi-label logits of shape (B, num_main_classes)
            aux: Dictionary containing auxiliary outputs (window mask, attention)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        raw_mask = make_padding_mask(x)

        # CNN encoding
        feat = self.encoder(x)
        enc_mask = self._downsample_mask_nearest(raw_mask, feat.shape[-1])

        # Window-based summarization
        h_win, m_win = self._make_windows(feat, enc_mask)

        # Project to Transformer dimension
        h_win = self.win_proj(h_win)

        # Global aggregation via Transformer [CLS] token
        bag, attn = self.transformer_pool(h_win, m_win)

        # Multi-label classification
        main_logits = self.cls_main(bag)

        aux = {"attention_pool": attn, "window_mask": m_win}
        return main_logits, aux