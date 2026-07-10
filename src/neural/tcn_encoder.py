"""
TCN (Temporal Convolutional Network) encoder for bank/UPI transaction sequences.

PDF: "Dilated causal TCN on raw debit/credit sequences -> 32-dim embedding."

Architecture: 3 stacked dilated-causal Conv1d blocks with residual connections.
Input shape : (batch, seq_len, 2)  -- [debit_amount, credit_amount] per day
Output shape: (batch, 32)          -- sequence embedding
"""
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

TCN_EMB_DIM = 32
TCN_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "tcn_encoder.pt"


class _DilatedCausalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3,
            padding=dilation * 2,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.dropout(self.relu(self.conv(x)))
        # Trim to match causal length
        out = out[:, :, : x.shape[-1]]
        return out + self.residual(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, emb_dim: int = TCN_EMB_DIM):
        super().__init__()
        self.blocks = nn.Sequential(
            _DilatedCausalBlock(input_dim, 32, dilation=1),
            _DilatedCausalBlock(32, 64, dilation=2),
            _DilatedCausalBlock(64, emb_dim, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return x


_tcn_model: TCNEncoder | None = None


def get_tcn_encoder() -> TCNEncoder:
    global _tcn_model
    if _tcn_model is None:
        _tcn_model = TCNEncoder()
        if TCN_MODEL_PATH.exists():
            _tcn_model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location="cpu"))
        _tcn_model.eval()
    return _tcn_model


def encode_transaction_sequence(daily_series: list[dict]) -> np.ndarray:
    """
    daily_series: list of {"debit": float, "credit": float} dicts (chronological).
    Returns 32-dim numpy embedding, or zeros if sequence is empty.
    """
    if not daily_series:
        return np.zeros(TCN_EMB_DIM, dtype=np.float32)

    arr = np.array(
        [[d.get("debit", 0.0), d.get("credit", 0.0)] for d in daily_series],
        dtype=np.float32,
    )
    # Normalise by max to avoid scale issues
    scale = arr.max() or 1.0
    arr = arr / scale

    tensor = torch.tensor(arr).unsqueeze(0)  # (1, seq_len, 2)
    with torch.no_grad():
        emb = get_tcn_encoder()(tensor).squeeze(0).numpy()
    return emb
