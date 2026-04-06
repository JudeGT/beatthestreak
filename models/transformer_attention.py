"""
HTL Model Component 2: Transformer Attention for Environmental Context.

Uses a single Transformer encoder block to capture interactions between:
  - Environmental/park features (air density, COR, park factor, weather)
  - Pitcher "deception" metrics (stuff+, tunnel consistency, archetype embeddings)

Multi-head attention allows the model to learn WHICH environmental factors
matter most for a given pitcher type (e.g., a finesse pitcher at Coors
is penalized less than a flyball pitcher at the same park).
"""

import torch
import torch.nn as nn
import math
from config import TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, DROPOUT


class EnvironmentAttention(nn.Module):
    """
    Transformer encoder over a flat set of environment + pitcher features.

    The input vector is split into 'tokens' by grouping features so the
    attention mechanism can learn cross-feature interactions. We use a
    simple token-per-feature-group approach:
      Token 0: Weather features     (temp_f, humidity_pct, pressure_mb, air_density)
      Token 1: Park features        (park_babip_factor, env_composite, altitude)
      Token 2: Pitcher stuff        (stuff_plus, velo, spin, tunnel_consistency)
      Token 3: Pitcher archetype    (8-dim one-hot)
      Token 4: Matchup context      (stand_enc, same_hand_matchup, cor_adjustment)

    Input
    -----
    env_feat : Tensor of shape (batch, n_env_features)
        Flat vector of environment + pitcher features from the Gold table.

    Output
    ------
    out : Tensor of shape (batch, d_model)
        Contextualized feature vector for fusion with LSTM output.
    """

    ENV_TOKEN_GROUPS = {
        "rolling":       17,   # ghp(5), h_pa(3), xwoba(3), barrel(2), velo(2), k/bb(2)
        "vulnerability": 4,    # pitcher hits allowed(2), k allowed(2)
        "stuff":         4,    # stuff_plus, velo, spin, tunnel
        "archetype":     8,    # one-hot pitcher archetypes
        "park":          8,    # babip_factor, env_composite, air_density, cor, humidity, temp, etc
        "matchup":       2,    # same_hand_matchup, stand_enc
    }
    N_TOKENS = len(ENV_TOKEN_GROUPS)

    def __init__(
        self,
        n_env_features: int = 43,   # total = 17 + 4 + 4 + 8 + 8 + 2
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.n_env_features = n_env_features
        self.d_model = d_model

        # Project each token group to d_model
        offsets = list(self.ENV_TOKEN_GROUPS.values())
        self.token_projs = nn.ModuleList([
            nn.Linear(sz, d_model) for sz in offsets
        ])

        # Positional encoding for 5 tokens
        self.pos_enc = nn.Parameter(
            torch.randn(1, self.N_TOKENS, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Final projection of mean-pooled tokens
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_tokens(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Split flat feature vector into token groups based on ENV_TOKEN_GROUPS.
        """
        tokens = []
        idx = 0
        for sz in self.ENV_TOKEN_GROUPS.values():
            tokens.append(x[:, idx:idx + sz])
            idx += sz
        return tokens

    def forward(self, env_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        env_feat : (batch, n_env_features)

        Returns
        -------
        out : (batch, d_model)
        """
        # Split into token groups and project each to d_model
        raw_tokens = self._split_tokens(env_feat)     # list of (batch, token_sz)
        projected  = [
            proj(tok).unsqueeze(1)                    # (batch, 1, d_model)
            for proj, tok in zip(self.token_projs, raw_tokens)
        ]
        # Stack: (batch, N_TOKENS, d_model)
        tokens = torch.cat(projected, dim=1) + self.pos_enc

        # Transformer encoding
        encoded = self.transformer(tokens)            # (batch, N_TOKENS, d_model)
        encoded = self.norm(encoded)

        # Mean pool over tokens → (batch, d_model)
        out = encoded.mean(dim=1)
        out = self.out_proj(out)
        return out

    @property
    def output_size(self) -> int:
        return self.d_model
