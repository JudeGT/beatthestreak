"""
HTL-Multimodal Model: Hybrid Transformer-LSTM (HTL).

Fuses:
  1. TemporalLSTM  — batter's last 100 PA sequence → hitting form context
  2. EnvironmentAttention — environment + pitcher features → situational context

Final prediction head:
  Linear → Sigmoid → P(Hit)  (trained with BCEWithLogitsLoss / Bernoulli loss)
"""

import torch
import torch.nn as nn
from models.lstm_temporal import TemporalLSTM
from models.transformer_attention import EnvironmentAttention
from config import DROPOUT


class HTLModel(nn.Module):
    """
    Hybrid Transformer-LSTM for hit probability prediction.

    Parameters
    ----------
    n_pa_features : int
        Number of features in each plate-appearance time step (LSTM input).
    n_env_features : int
        Number of flat environment/pitcher features (Transformer input).
    lstm_hidden : int
        Hidden size for the TemporalLSTM (default from config).
    transformer_d : int
        d_model for EnvironmentAttention (default from config).
    dropout : float
        Dropout rate used throughout.

    Forward inputs
    --------------
    pa_seq : Tensor (batch, seq_len, n_pa_features)
        Temporal plate-appearance sequence for each batter.
    env_feat : Tensor (batch, n_env_features)
        Flat environment + pitcher feature vector.

    Forward output
    --------------
    logit : Tensor (batch,)
        Raw logit. Apply sigmoid for P(Hit). Use BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        n_pa_features: int = 8,
        n_env_features: int = 22,
        lstm_hidden: int = 256,
        transformer_d: int = 128,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.lstm = TemporalLSTM(
            n_pa_features=n_pa_features,
            hidden_size=lstm_hidden,
            dropout=dropout,
        )

        self.transformer = EnvironmentAttention(
            n_env_features=n_env_features,
            d_model=transformer_d,
            dropout=dropout,
        )

        lstm_out_size = self.lstm.output_size          # hidden_size * 2
        env_out_size  = self.transformer.output_size   # transformer_d

        fusion_in = lstm_out_size + env_out_size

        # Learned fusion attention gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_in, fusion_in),
            nn.Sigmoid(),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pa_seq: torch.Tensor,
        env_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pa_seq  : (batch, seq_len, n_pa_features)
        env_feat: (batch, n_env_features)

        Returns
        -------
        logit : (batch,)
        """
        lstm_ctx = self.lstm(pa_seq)              # (batch, lstm_hidden*2)
        env_ctx  = self.transformer(env_feat)     # (batch, transformer_d)

        # Concatenate modalities
        fused = torch.cat([lstm_ctx, env_ctx], dim=-1)   # (batch, fusion_in)

        # Gated fusion
        gate  = self.fusion_gate(fused)
        fused = fused * gate

        logit = self.head(fused).squeeze(-1)   # (batch,)
        return logit

    def predict_prob(
        self,
        pa_seq: torch.Tensor,
        env_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid probability P(Hit) ∈ [0, 1]."""
        with torch.no_grad():
            logit = self.forward(pa_seq, env_feat)
            return torch.sigmoid(logit)
