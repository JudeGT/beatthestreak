"""
HTL Model Component 1: Temporal LSTM.

Processes a sequence of a batter's last 100 plate appearances (temporal context)
using a 2-layer bidirectional LSTM. Outputs a fixed-size hidden state that
summarizes recent hitting form, fatigue signals, and slump/streak patterns.
"""

import torch
import torch.nn as nn
from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_SEQUENCE_LEN, DROPOUT


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM encoding of the last N plate appearances.

    Input
    -----
    x : Tensor of shape (batch, seq_len, n_pa_features)
        Time-ordered sequence of plate-appearance features per batter.
        Typical PA features: [exit_velo, launch_angle, xba, xwoba, barrel_flag,
                               pitch_type_enc, pitcher_hand_enc, result_enc]

    Output
    ------
    h : Tensor of shape (batch, hidden_size * 2)
        Final bidirectional hidden state (forward + backward concatenated).
    """

    def __init__(
        self,
        n_pa_features: int = 8,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = DROPOUT,
        seq_len: int = LSTM_SEQUENCE_LEN,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.seq_len     = seq_len

        # Input projection: map raw PA features to LSTM input dimension
        self.input_proj = nn.Sequential(
            nn.Linear(n_pa_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Self-attention over LSTM outputs (temporal pooling)
        self.attn_proj = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_pa_features)

        Returns
        -------
        ctx : (batch, hidden_size * 2)
            Attention-weighted summary of the PA sequence.
        """
        # (batch, seq_len, hidden_size)
        x_proj = self.input_proj(x)

        # (batch, seq_len, hidden_size*2)
        lstm_out, _ = self.lstm(x_proj)
        lstm_out = self.dropout(lstm_out)

        # Attention pooling over time steps
        attn_scores = self.attn_proj(lstm_out).squeeze(-1)   # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Weighted sum: (batch, hidden_size*2)
        ctx = (lstm_out * attn_weights).sum(dim=1)
        return ctx

    @property
    def output_size(self) -> int:
        return self.hidden_size * 2  # bidirectional
