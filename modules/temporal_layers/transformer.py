import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# modules/temporal_layers.py

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Produces [1, T, C] that you can add to x [B, T, C].
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, C]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # [C/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even idx
        pe[:, 1::2] = torch.cos(position * div_term)  # odd idx
        pe = pe.unsqueeze(0)  # [1, max_len, C]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        T = x.size(1)
        return self.pe[:, :T, :]


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder over TIME dimension.
    Input : x [B, T, C], feat_len [B]
    Output: y [B, T, C] (same shape)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward= None,
        dropout: float = 0.1,
        use_sinusoidal_pos_emb: bool = True,
        max_len: int = 5000,
        pre_norm: bool = True,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Positional embedding
        if use_sinusoidal_pos_emb:
            self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        else:
            # Learnable positional embeddings
            self.pos_emb = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,     # IMPORTANT: keep [B, T, C]
            norm_first=pre_norm,  # Pre-norm tends to be stabler
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.use_sinusoidal = use_sinusoidal_pos_emb
        self.max_len = max_len

    @staticmethod
    def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        """
        lengths: [B] (can be on CPU), max_len: int
        Returns mask [B, max_len] on `device` with True for padding positions.
        """
        lengths = lengths.to(device=device, dtype=torch.long)
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # [1, T]
        mask = idx >= lengths.unsqueeze(1)                       # [B, T]
        return mask


    def forward(self, x: torch.Tensor, feat_len: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        feat_len: [B]
        """
        assert x.dim() == 3, f"Expected x [B,T,C], got {x.shape}"
        B, T, C = x.shape

        # Build positional encoding
        if self.use_sinusoidal:
            pos = self.pos_emb(x)  # [1, T, C]
            x = x + pos
        else:
            if T > self.max_len:
                raise ValueError(f"T={T} exceeds max_len={self.max_len} for learnable pos_emb.")
            positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B, T]
            x = x + self.pos_emb(positions)  # [B, T, C]

        x = self.dropout(x)

        # Key padding mask: True means "ignore"
        src_key_padding_mask = self.lengths_to_padding_mask(feat_len, T, device=x.device)  # [B, T]

        # Transformer encoder
        y = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, C]

        return y
